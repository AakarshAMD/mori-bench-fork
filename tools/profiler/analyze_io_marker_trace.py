#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Analyze a MORI-IO rocprofv3 *marker* trace (roctx ranges), NOT a kernel trace.

Why this exists
---------------
A MoRI RDMA write/read is a HOST-side ``ibv_post_send`` + NIC hardware operation,
NOT a HIP kernel.  So ``rocprofv3 --kernel-trace`` is effectively BLIND to the
transfer path (for ``--mem-type cpu`` the kernel trace is empty).  The MoRI-IO
fork instead emits its own roctx ranges (see ``src/io/roctx_mori.hpp``), captured
with ``rocprofv3 --marker-trace`` + the SDK roctx lib
``librocprofiler-sdk-roctx.so``.  Enable them via ``MORI_ROCTX=1`` (synchronous
host-post anchors) and ``MORI_ROCTX_TRANSFER=1`` (async post->CQ wire ranges).

Ranges consumed
---------------
  * ``mori.rdma.kv_transfer``        write, async post->CQE (real wire duration)
  * ``mori.rdma.kv_transfer.read``   read,  async post->CQE
  * ``mori.rdma.batch_post.{write,read}``  synchronous host-post window
  * ``mori.io.engine_batch_write``         synchronous whole-call host-post window
Each kv_transfer range name is tagged ``bytes=<N> wrs=<M> merged=<K> id=<id>``.

What it answers (the user's questions)
--------------------------------------
  1. IDLE  -- gaps between consecutive kv_transfer ranges (NIC starved) and the
     idle fraction of the capture window not covered by any in-flight transfer.
  2. OVERLAP -- max/avg number of temporally-overlapping kv_transfer ranges =
     achieved in-flight pipeline depth (compare to num-qp-per-transfer / batch).
  3. Host-post vs wire: time in batch_post ranges and gaps between them.

Input
-----
Point ``trace`` at either a rocprofv3 output directory (the tool finds
``*_marker_api_trace.csv``) or a specific marker CSV/JSON file.  Timestamps in
rocprofv3 CSVs are nanoseconds by default (override with --time-unit).

Outputs a human summary to stdout and, with --out-csv, a derived-metrics CSV.
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict

# Range-name substrings we care about.
KV_WRITE = "mori.rdma.kv_transfer"
KV_READ = "mori.rdma.kv_transfer.read"
POST_PREFIXES = ("mori.rdma.batch_post", "mori.io.engine_batch_write")

# bytes=/wrs=/merged=/id= are appended by roctx_mori.hpp (id= is always LAST).
_TAG_RE = {
    "bytes": re.compile(r"bytes=(\d+)"),
    "wrs": re.compile(r"wrs=(\d+)"),
    "merged": re.compile(r"merged=(\d+)"),
    "id": re.compile(r"id=(\d+)"),
}


class Range:
    __slots__ = ("name", "base", "start", "end", "tid", "bytes", "wrs", "merged", "xid")

    def __init__(self, name, start, end, tid):
        self.name = name
        self.start = start  # microseconds (normalized)
        self.end = end
        self.tid = tid
        self.base = _base_name(name)
        self.bytes = _tag(name, "bytes")
        self.wrs = _tag(name, "wrs")
        self.merged = _tag(name, "merged")
        self.xid = _tag(name, "id")

    @property
    def dur(self):
        return self.end - self.start


def _tag(name, key):
    m = _TAG_RE[key].search(name)
    return int(m.group(1)) if m else None


def _base_name(name):
    """Strip the ' bytes=... id=...' suffix to get the bare range name."""
    return name.split(" ", 1)[0].strip()


# ---------------------------------------------------------------------------
# Percentile / stats helpers (stdlib only)
# ---------------------------------------------------------------------------


def _percentile(sorted_vals, pct):
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return sorted_vals[lo]
    frac = rank - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _stats(vals):
    """Return dict of count/min/avg/p50/p90/p99/max/std for a list of floats."""
    if not vals:
        return dict(count=0, min=0, avg=0, p50=0, p90=0, p99=0, max=0, std=0, total=0)
    s = sorted(vals)
    n = len(s)
    avg = sum(s) / n
    std = math.sqrt(sum((x - avg) ** 2 for x in s) / (n - 1)) if n > 1 else 0.0
    return dict(
        count=n,
        min=s[0],
        avg=avg,
        p50=_percentile(s, 50),
        p90=_percentile(s, 90),
        p99=_percentile(s, 99),
        max=s[-1],
        std=std,
        total=sum(s),
    )


# ---------------------------------------------------------------------------
# Interval-set helpers (overlap depth + union/idle)
# ---------------------------------------------------------------------------


def union_busy(intervals):
    """Total wall time covered by the union of [start,end) intervals."""
    if not intervals:
        return 0.0
    s = sorted(intervals)
    busy = 0.0
    cs, ce = s[0]
    for a, b in s[1:]:
        if a <= ce:
            ce = max(ce, b)
        else:
            busy += ce - cs
            cs, ce = a, b
    busy += ce - cs
    return busy


def overlap_depth(intervals):
    """Sweep-line concurrency. Returns (max_depth, time_weighted_avg_depth,
    time_weighted_avg_depth_while_busy)."""
    if not intervals:
        return 0, 0.0, 0.0
    events = []
    for a, b in intervals:
        if b < a:
            a, b = b, a
        events.append((a, 1))
        events.append((b, -1))
    events.sort(key=lambda e: (e[0], -e[1]))  # opens before closes at same ts
    depth = 0
    max_depth = 0
    prev_t = events[0][0]
    area = 0.0          # depth integrated over all time
    busy_area = 0.0     # depth integrated over time where depth>0
    busy_time = 0.0
    for t, delta in events:
        dt = t - prev_t
        if dt > 0:
            area += depth * dt
            if depth > 0:
                busy_area += depth * dt
                busy_time += dt
        depth += delta
        max_depth = max(max_depth, depth)
        prev_t = t
    span = events[-1][0] - events[0][0]
    avg_depth = area / span if span > 0 else 0.0
    avg_depth_busy = busy_area / busy_time if busy_time > 0 else 0.0
    return max_depth, avg_depth, avg_depth_busy


def gaps_between(intervals):
    """Idle gaps between consecutive intervals ordered by start (running max end).
    Returns list of positive gap durations."""
    if len(intervals) < 2:
        return []
    s = sorted(intervals)
    gaps = []
    cur_end = s[0][1]
    for a, b in s[1:]:
        if a > cur_end:
            gaps.append(a - cur_end)
        cur_end = max(cur_end, b)
    return gaps


# ---------------------------------------------------------------------------
# Parsing: rocprofv3 marker CSV (primary) + JSON (best-effort)
# ---------------------------------------------------------------------------

_NAME_COLS = ["function", "name", "operation", "message", "kind", "label"]
_START_COLS = ["start_timestamp", "start", "start_ns", "begin", "begin_timestamp"]
_END_COLS = ["end_timestamp", "end", "end_ns", "finish"]
_TID_COLS = ["thread_id", "tid"]


def _find_col(header_lower, candidates):
    for c in candidates:
        if c in header_lower:
            return header_lower.index(c)
    return None


def _time_scale(unit):
    return {"ns": 1e-3, "us": 1.0, "ms": 1e3, "s": 1e6}[unit]  # -> microseconds


def parse_marker_csv(path, unit):
    scale = _time_scale(unit)
    ranges = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return ranges
        hl = [h.strip().lower() for h in header]
        ni = _find_col(hl, _NAME_COLS)
        si = _find_col(hl, _START_COLS)
        ei = _find_col(hl, _END_COLS)
        ti = _find_col(hl, _TID_COLS)
        if ni is None or si is None or ei is None:
            # Some rocprofv3 builds put the roctx message in the LAST column.
            if ni is None:
                ni = len(hl) - 1
            if si is None or ei is None:
                raise ValueError(
                    f"{path}: could not locate start/end timestamp columns in "
                    f"header {header!r}"
                )
        for row in reader:
            if not row or len(row) <= max(ni, si, ei):
                continue
            name = row[ni].strip().strip('"')
            if not _is_interesting(name):
                continue
            try:
                start = float(row[si]) * scale
                end = float(row[ei]) * scale
            except ValueError:
                continue
            tid = row[ti].strip() if ti is not None and ti < len(row) else "?"
            ranges.append(Range(name, start, end, tid))
    return ranges


def parse_marker_json(path, unit):
    """Best-effort JSON parse. Handles (a) chrome-trace traceEvents (ph B/E) and
    (b) rocprofv3 SDK json with a marker buffer + strings table."""
    scale = _time_scale(unit)
    with open(path) as f:
        data = json.load(f)
    ranges = []

    # (a) chrome trace style
    if isinstance(data, dict) and "traceEvents" in data:
        by_tid = defaultdict(lambda: {"B": [], "E": []})
        for e in data["traceEvents"]:
            ph = e.get("ph")
            if ph in ("B", "E"):
                by_tid[e.get("tid", 0)][ph].append(e)
            elif ph == "X" and _is_interesting(e.get("name", "")):
                s = float(e["ts"]) * scale
                ranges.append(
                    Range(e["name"], s, s + float(e.get("dur", 0)) * scale, e.get("tid", "?"))
                )
        for tid, phases in by_tid.items():
            begins = sorted(phases["B"], key=lambda x: x["ts"])
            ends = sorted(phases["E"], key=lambda x: x["ts"])
            for b, e in zip(begins, ends):
                if _is_interesting(b.get("name", "")):
                    ranges.append(
                        Range(b["name"], float(b["ts"]) * scale, float(e["ts"]) * scale, tid)
                    )
        return ranges

    # (b) rocprofv3 SDK json (schema varies; scan generically for marker records)
    def _walk(obj):
        if isinstance(obj, dict):
            # A record with a message/name + start/end timestamps
            nm = obj.get("message") or obj.get("name")
            st = obj.get("start_timestamp") or obj.get("start")
            en = obj.get("end_timestamp") or obj.get("end")
            if isinstance(nm, str) and st is not None and en is not None and _is_interesting(nm):
                try:
                    ranges.append(
                        Range(nm, float(st) * scale, float(en) * scale, obj.get("thread_id", "?"))
                    )
                except (TypeError, ValueError):
                    pass
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for v in obj:
                _walk(v)

    _walk(data)
    return ranges


def _is_interesting(name):
    return name.startswith(KV_WRITE) or name.startswith(POST_PREFIXES)


def _discover_inputs(path):
    """Return a list of (path, kind) inputs. kind in {'csv','json'}."""
    if os.path.isdir(path):
        csvs = sorted(glob.glob(os.path.join(path, "*marker_api_trace.csv")))
        if not csvs:
            csvs = sorted(
                p
                for p in glob.glob(os.path.join(path, "*.csv"))
                if "marker" in os.path.basename(p).lower()
            )
        if csvs:
            return [(p, "csv") for p in csvs]
        jsons = sorted(glob.glob(os.path.join(path, "*.json")))
        return [(p, "json") for p in jsons]
    ext = os.path.splitext(path)[1].lower()
    return [(path, "json" if ext == ".json" else "csv")]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_stat(label, st, unit="us"):
    return (
        f"  {label:<26} n={st['count']:<6} "
        f"min={st['min']:.2f} avg={st['avg']:.2f} p50={st['p50']:.2f} "
        f"p90={st['p90']:.2f} p99={st['p99']:.2f} max={st['max']:.2f} "
        f"std={st['std']:.2f} ({unit})"
    )


def analyze(ranges, out_csv=None, line_rate_gbps=0.0):
    kv = [r for r in ranges if r.base in (KV_WRITE, KV_READ)]
    kv_write = [r for r in kv if r.base == KV_WRITE]
    kv_read = [r for r in kv if r.base == KV_READ]
    posts = [r for r in ranges if any(r.base.startswith(p) for p in POST_PREFIXES)]

    print("=" * 78)
    print("MORI-IO MARKER-TRACE ANALYSIS")
    print("=" * 78)
    print(
        f"parsed ranges: {len(ranges)}  "
        f"(kv_transfer write={len(kv_write)} read={len(kv_read)}; "
        f"host-post={len(posts)})"
    )
    if not kv and not posts:
        print(
            "\n[!] No MoRI-IO roctx ranges found. Check that the run had "
            "MORI_ROCTX_TRANSFER=1 / MORI_ROCTX=1 and that rocprofv3 --marker-trace "
            "captured librocprofiler-sdk-roctx.so (NOT legacy libroctx64)."
        )
        return

    derived = {}

    # ---- Per-transfer duration stats (wire duration = post->CQE) -------------
    if kv:
        print("\n[1] PER-TRANSFER WIRE DURATION (async post->CQE ranges)")
        for label, group in (("kv_transfer(write)", kv_write), ("kv_transfer.read", kv_read)):
            if group:
                st = _stats([r.dur for r in group])
                print(_fmt_stat(label, st))
                derived[f"{label}.count"] = st["count"]
                derived[f"{label}.dur_avg_us"] = round(st["avg"], 3)
                derived[f"{label}.dur_p50_us"] = round(st["p50"], 3)
                derived[f"{label}.dur_p99_us"] = round(st["p99"], 3)
                derived[f"{label}.dur_max_us"] = round(st["max"], 3)
        # byte-weighted achieved BW per transfer, if bytes are tagged
        byted = [r for r in kv if r.bytes]
        if byted:
            bws = [
                (r.bytes / 1e9) / (r.dur / 1e6)
                for r in byted
                if r.dur > 0
            ]  # GB/s
            if bws:
                bst = _stats(bws)
                print(
                    f"  per-transfer BW (bytes/dur):  "
                    f"min={bst['min']:.2f} avg={bst['avg']:.2f} max={bst['max']:.2f} GB/s"
                )
                derived["per_transfer_bw_avg_gbps"] = round(bst["avg"], 3)
                derived["per_transfer_bw_max_gbps"] = round(bst["max"], 3)

    # ---- Overlap / in-flight pipeline depth ---------------------------------
    if kv:
        ivals = [(r.start, r.end) for r in kv]
        max_d, avg_d, avg_d_busy = overlap_depth(ivals)
        print("\n[2] OVERLAP / IN-FLIGHT PIPELINE DEPTH (concurrent kv_transfer ranges)")
        print(f"  max concurrent in-flight transfers : {max_d}")
        print(f"  avg concurrent (over whole capture): {avg_d:.3f}")
        print(f"  avg concurrent (while >=1 in-flight): {avg_d_busy:.3f}")
        if max_d <= 1:
            print(
                "  -> NO overlap: transfers are effectively serialized "
                "(each completes before the next is in flight)."
            )
        derived["overlap_max_depth"] = max_d
        derived["overlap_avg_depth"] = round(avg_d, 4)
        derived["overlap_avg_depth_busy"] = round(avg_d_busy, 4)

    # ---- Idle gaps + busy/idle fraction over the capture window -------------
    if kv:
        ivals = [(r.start, r.end) for r in kv]
        win_start = min(a for a, _ in ivals)
        win_end = max(b for _, b in ivals)
        window = win_end - win_start
        busy = union_busy(ivals)
        idle = max(0.0, window - busy)
        gaps = gaps_between(ivals)
        print("\n[3] IDLE / GAPS (transfer timeline)")
        print(f"  capture window (first post -> last CQE): {window:.2f} us")
        print(
            f"  wire-busy (union of in-flight): {busy:.2f} us "
            f"({100.0 * busy / window if window else 0:.1f}%)"
        )
        print(
            f"  idle (no transfer in flight)  : {idle:.2f} us "
            f"({100.0 * idle / window if window else 0:.1f}%)"
        )
        if gaps:
            gst = _stats(gaps)
            print(
                f"  inter-transfer gaps: n={gst['count']} total={gst['total']:.2f}us "
                f"avg={gst['avg']:.2f} p90={gst['p90']:.2f} max={gst['max']:.2f} (us)"
            )
            derived["idle_gap_count"] = gst["count"]
            derived["idle_gap_total_us"] = round(gst["total"], 3)
            derived["idle_gap_avg_us"] = round(gst["avg"], 3)
            derived["idle_gap_max_us"] = round(gst["max"], 3)
        else:
            print("  inter-transfer gaps: none (transfers overlap or are contiguous)")
        derived["capture_window_us"] = round(window, 3)
        derived["wire_busy_us"] = round(busy, 3)
        derived["wire_busy_frac"] = round(busy / window, 4) if window else 0.0
        derived["idle_us"] = round(idle, 3)
        derived["idle_frac"] = round(idle / window, 4) if window else 0.0

    # ---- Host-post windows (synchronous MORI_ROCTX anchors) ------------------
    if posts:
        pst = _stats([r.dur for r in posts])
        p_ivals = [(r.start, r.end) for r in posts]
        p_gaps = gaps_between(p_ivals)
        print("\n[4] HOST-POST WINDOWS (synchronous; building WRs + doorbell)")
        print(_fmt_stat("host_post", pst))
        if p_gaps:
            pgst = _stats(p_gaps)
            print(
                f"  gaps between host-post windows (host not feeding NIC): "
                f"n={pgst['count']} total={pgst['total']:.2f}us avg={pgst['avg']:.2f} "
                f"max={pgst['max']:.2f} (us)"
            )
            derived["host_post_gap_total_us"] = round(pgst["total"], 3)
            derived["host_post_gap_avg_us"] = round(pgst["avg"], 3)
        derived["host_post_avg_us"] = round(pst["avg"], 3)
        derived["host_post_count"] = pst["count"]

    # ---- Post->completion coupling (if we have both lanes) ------------------
    # For each host-post window, the tail gap to the next kv_transfer completion
    # approximates the post->wire latency the host cannot see in a single timer.
    if posts and kv:
        kv_ends = sorted(r.end for r in kv)
        print("\n[5] INTERPRETATION")
        print(
            "  host-post time is the SUBMIT cost; kv_transfer duration is the "
            "post->CQE WIRE cost. If wire >> host-post, the host is NOT the "
            "bottleneck (NIC/link bound); if idle_frac is high with low overlap, "
            "the pipeline is starved (increase batch / QP / in-flight depth)."
        )

    # ---- Derived-metrics CSV -------------------------------------------------
    if out_csv:
        if line_rate_gbps > 0 and "per_transfer_bw_max_gbps" in derived:
            # per_transfer_bw is GB/s (bytes); *8 -> Gb/s (bits) vs line rate.
            derived["eff_vs_line_pct"] = round(
                100.0 * (derived["per_transfer_bw_max_gbps"] * 8.0) / line_rate_gbps, 2
            )
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            for k, v in derived.items():
                w.writerow([k, v])
        print(f"\n[saved] derived metrics CSV -> {out_csv} ({len(derived)} metrics)")
    print("=" * 78)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "trace",
        help="rocprofv3 output dir (auto-finds *_marker_api_trace.csv) OR a "
        "specific marker CSV/JSON file.",
    )
    p.add_argument(
        "--time-unit",
        choices=["ns", "us", "ms", "s"],
        default="ns",
        help="Unit of the trace timestamps (rocprofv3 CSV is ns; default: ns).",
    )
    p.add_argument(
        "--out-csv",
        default="",
        help="Write derived idle/overlap/duration metrics to this CSV path.",
    )
    p.add_argument(
        "--line-rate-gbps",
        type=float,
        default=0.0,
        help="Optional line rate (Gb/s) for an efficiency %% in the derived CSV.",
    )
    args = p.parse_args()

    inputs = _discover_inputs(args.trace)
    if not inputs:
        print(f"[!] no marker CSV/JSON found under {args.trace!r}", file=sys.stderr)
        sys.exit(2)

    all_ranges = []
    for path, kind in inputs:
        print(f"[loading] {path}  (kind={kind}, time-unit={args.time_unit})")
        try:
            if kind == "csv":
                all_ranges.extend(parse_marker_csv(path, args.time_unit))
            else:
                all_ranges.extend(parse_marker_json(path, args.time_unit))
        except Exception as e:  # noqa: BLE001 - surface parse issues, keep going
            print(f"  [!] failed to parse {path}: {e}", file=sys.stderr)

    analyze(all_ranges, out_csv=args.out_csv or None, line_rate_gbps=args.line_rate_gbps)


if __name__ == "__main__":
    main()
