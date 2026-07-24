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
Each kv_transfer range name is tagged
``bytes=<N> wrs=<M> merged=<K> qp=<Q> id=<id>``.

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
QP_POST = "mori.rdma.qp_post"
QP_CQE = "mori.rdma.qp_cqe"
PHASE = "mori.bench.phase"
POST_PREFIXES = ("mori.rdma.batch_post", "mori.io.engine_batch_write")

# bytes=/wrs=/merged=/qp=/id= are appended by roctx_mori.hpp (id= is always LAST).
_TAG_RE = {
    "bytes": re.compile(r"bytes=(\d+)"),
    "wrs": re.compile(r"wrs=(\d+)"),
    "merged": re.compile(r"merged=(\d+)"),
    "qp": re.compile(r"qp=(\d+)"),
    "id": re.compile(r"id=(\d+)"),
    "key": re.compile(r"key=(\d+)"),
}
_OP_RE = re.compile(r"op=(read|write)")


class Range:
    __slots__ = (
        "name", "base", "start", "end", "tid", "pid", "source",
        "bytes", "wrs", "merged", "qp", "xid", "key", "op",
    )

    def __init__(self, name, start, end, tid, pid="?", source=""):
        self.name = name
        self.start = start  # microseconds (normalized)
        self.end = end
        self.tid = tid
        self.pid = str(pid)
        self.source = source
        self.base = _base_name(name)
        self.bytes = _tag(name, "bytes")
        self.wrs = _tag(name, "wrs")
        self.merged = _tag(name, "merged")
        self.qp = _tag(name, "qp")
        self.xid = _tag(name, "id")
        self.key = _tag(name, "key")
        op = _OP_RE.search(name)
        self.op = op.group(1) if op else ("read" if self.base == KV_READ else "write")

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
_PID_COLS = ["process_id", "pid"]


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
        pi = _find_col(hl, _PID_COLS)
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
            pid = row[pi].strip() if pi is not None and pi < len(row) else "?"
            ranges.append(Range(name, start, end, tid, pid, os.path.abspath(path)))
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
                    Range(
                        e["name"], s, s + float(e.get("dur", 0)) * scale,
                        e.get("tid", "?"), e.get("pid", "?"), os.path.abspath(path)
                    )
                )
        for tid, phases in by_tid.items():
            begins = sorted(phases["B"], key=lambda x: x["ts"])
            ends = sorted(phases["E"], key=lambda x: x["ts"])
            for b, e in zip(begins, ends):
                if _is_interesting(b.get("name", "")):
                    ranges.append(
                        Range(
                            b["name"], float(b["ts"]) * scale, float(e["ts"]) * scale,
                            tid, b.get("pid", "?"), os.path.abspath(path)
                        )
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
                        Range(
                            nm, float(st) * scale, float(en) * scale,
                            obj.get("thread_id", "?"), obj.get("process_id", "?"),
                            os.path.abspath(path)
                        )
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
    return (
        name.startswith(KV_WRITE)
        or name.startswith(QP_POST)
        or name.startswith(QP_CQE)
        or name.startswith(PHASE)
        or name.startswith(POST_PREFIXES)
    )


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
# Keyed instant-event reconstruction
# ---------------------------------------------------------------------------


class QpStripe:
    __slots__ = (
        "source", "pid", "key", "xid", "qp", "op", "bytes", "wrs", "merged",
        "post", "cqe",
    )

    def __init__(self, post, cqe):
        self.source = post.source
        self.pid = post.pid
        self.key = post.key
        self.xid = post.xid
        self.qp = post.qp
        self.op = post.op
        self.bytes = post.bytes
        self.wrs = post.wrs
        self.merged = post.merged
        self.post = post.start
        self.cqe = cqe.start

    @property
    def dur(self):
        return self.cqe - self.post

    @property
    def bw(self):
        return (self.bytes / 1e9) / (self.dur / 1e6) if self.bytes and self.dur > 0 else 0.0


def _phase_windows(records):
    """Return per-process measured windows and phase-boundary diagnostics."""
    bounds = defaultdict(lambda: {"begin": [], "end": []})
    for r in records:
        if r.base != PHASE:
            continue
        if "measured_begin" in r.name:
            bounds[(r.source, r.pid)]["begin"].append(r.start)
        elif "measured_end" in r.name:
            bounds[(r.source, r.pid)]["end"].append(r.start)
    windows = defaultdict(list)
    issues = []
    for proc, sides in bounds.items():
        begins = sorted(sides["begin"])
        ends = sorted(sides["end"])
        if len(begins) != len(ends):
            issues.append(
                f"phase boundary mismatch source={os.path.basename(proc[0])} pid={proc[1]} "
                f"begin={len(begins)} end={len(ends)}"
            )
        for begin, end in zip(begins, ends):
            if end >= begin:
                windows[proc].append((begin, end))
            else:
                issues.append(
                    f"phase end precedes begin source={os.path.basename(proc[0])} pid={proc[1]}"
                )
    return windows, issues


def _in_measured_phase(record, windows):
    proc_windows = windows.get((record.source, record.pid))
    if not proc_windows:
        return True
    return any(begin <= record.start <= end for begin, end in proc_windows)


def join_qp_events(records, measured_only=True):
    """Join qp_post -> qp_cqe using (trace source, process, key, qp).

    Returns (stripes, diagnostics, used_phase_filter). Duplicate and missing
    endpoints are diagnostics and are never paired by order.
    """
    windows, diagnostics = _phase_windows(records)
    events = [
        r
        for r in records
        if r.base in (QP_POST, QP_CQE)
        and (not measured_only or _in_measured_phase(r, windows))
    ]
    grouped = defaultdict(lambda: {"post": [], "cqe": []})
    for event in events:
        if event.key is None or event.qp is None:
            diagnostics.append(f"unkeyed {event.base}: {event.name}")
            continue
        side = "post" if event.base == QP_POST else "cqe"
        grouped[(event.source, event.pid, event.key, event.qp)][side].append(event)

    stripes = []
    for join_key, sides in sorted(grouped.items(), key=lambda item: str(item[0])):
        posts, cqes = sides["post"], sides["cqe"]
        short = (
            f"source={os.path.basename(join_key[0])} pid={join_key[1]} "
            f"key={join_key[2]} qp={join_key[3]}"
        )
        if len(posts) != 1 or len(cqes) != 1:
            diagnostics.append(f"endpoint cardinality {short} post={len(posts)} cqe={len(cqes)}")
            continue
        post, cqe = posts[0], cqes[0]
        for field in ("xid", "qp", "bytes", "wrs", "merged", "op"):
            if getattr(post, field) != getattr(cqe, field):
                diagnostics.append(
                    f"metadata mismatch {short} field={field} "
                    f"post={getattr(post, field)} cqe={getattr(cqe, field)}"
                )
        stripe = QpStripe(post, cqe)
        if stripe.dur < 0:
            diagnostics.append(f"negative duration {short} duration_us={stripe.dur:.3f}")
            continue
        stripes.append(stripe)
    return stripes, diagnostics, bool(windows)


def group_logical_transfers(stripes):
    groups = defaultdict(list)
    for stripe in stripes:
        groups[(stripe.source, stripe.pid, stripe.xid, stripe.op)].append(stripe)
    logical = []
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        start = min(s.post for s in group)
        end = max(s.cqe for s in group)
        total_bytes = sum(s.bytes or 0 for s in group)
        qps = sorted({s.qp for s in group})
        logical.append(
            {
                "source": key[0],
                "pid": key[1],
                "id": key[2],
                "op": key[3],
                "stripes": group,
                "qps": qps,
                "bytes": total_bytes,
                "start": start,
                "end": end,
                "duration_us": end - start,
                "bw_gb_s": (total_bytes / 1e9) / ((end - start) / 1e6) if end > start else 0.0,
                "start_skew_us": max(s.post for s in group) - start,
                "completion_skew_us": end - min(s.cqe for s in group),
            }
        )
    return logical


def write_detail_csv(path, rows, fields):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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


def analyze(
    ranges,
    out_csv=None,
    line_rate_gbps=0.0,
    stripes_csv=None,
    logical_csv=None,
    measured_only=True,
):
    kv = [r for r in ranges if r.base in (KV_WRITE, KV_READ)]
    kv_write = [r for r in kv if r.base == KV_WRITE]
    kv_read = [r for r in kv if r.base == KV_READ]
    posts = [r for r in ranges if any(r.base.startswith(p) for p in POST_PREFIXES)]
    endpoint_events = [r for r in ranges if r.base in (QP_POST, QP_CQE)]

    print("=" * 78)
    print("MORI-IO MARKER-TRACE ANALYSIS")
    print("=" * 78)
    print(
        f"parsed records: {len(ranges)}  keyed endpoints={len(endpoint_events)}  "
        f"(kv_transfer write={len(kv_write)} read={len(kv_read)}; "
        f"host-post={len(posts)})"
    )
    if not endpoint_events and not kv and not posts:
        print(
            "\n[!] No MoRI-IO roctx ranges found. Check that the run had "
            "MORI_ROCTX_TRANSFER=1 / MORI_ROCTX=1 and that rocprofv3 --marker-trace "
            "captured librocprofiler-sdk-roctx.so (NOT legacy libroctx64)."
        )
        return

    derived = {}

    stripes = []
    if endpoint_events:
        stripes, diagnostics, used_phase_filter = join_qp_events(ranges, measured_only)
        logical = group_logical_transfers(stripes)
        windows, _ = _phase_windows(ranges)
        selected_endpoints = [
            event
            for event in endpoint_events
            if not measured_only or _in_measured_phase(event, windows)
        ]
        post_count = sum(r.base == QP_POST for r in selected_endpoints)
        cqe_count = sum(r.base == QP_CQE for r in selected_endpoints)
        phase_selection = (
            "explicit measured windows"
            if used_phase_filter and measured_only
            else "all records; no measured-phase filtering available"
        )
        print("\n[1] KEYED PER-QP STRIPES (qp_post -> qp_cqe instant marks)")
        print(f"  phase selection: {phase_selection}")
        print(
            f"  selected endpoints: post={post_count} cqe={cqe_count} "
            f"joined per-QP stripes={len(stripes)}"
        )
        for stripe in sorted(stripes, key=lambda s: (s.source, s.pid, s.xid, s.qp, s.key)):
            print(
                f"  per-QP stripe id={stripe.xid} qp={stripe.qp} key={stripe.key} "
                f"bytes={stripe.bytes} wrs={stripe.wrs} duration={stripe.dur:.3f} us "
                f"stripe_bw={stripe.bw:.3f} GB/s"
            )
        if diagnostics:
            print(f"  [!] endpoint diagnostics ({len(diagnostics)}):")
            for issue in diagnostics:
                print(f"      {issue}")
        else:
            print("  endpoint diagnostics: none (all keys unique and complete)")

        print("\n[2] LOGICAL TRANSFERS (all per-QP stripes grouped by process/id/op)")
        for item in logical:
            print(
                f"  logical id={item['id']} op={item['op']} qps={item['qps']} "
                f"stripes={len(item['stripes'])} bytes={item['bytes']} "
                f"duration={item['duration_us']:.3f} us bw={item['bw_gb_s']:.3f} GB/s "
                f"start_skew={item['start_skew_us']:.3f} us "
                f"completion_skew={item['completion_skew_us']:.3f} us"
            )
        derived["keyed_endpoint_post_count"] = post_count
        derived["keyed_endpoint_cqe_count"] = cqe_count
        derived["keyed_endpoint_raw_count"] = len(endpoint_events)
        derived["keyed_joined_stripe_count"] = len(stripes)
        derived["keyed_diagnostic_count"] = len(diagnostics)
        derived["logical_transfer_count"] = len(logical)
        derived["measured_phase_filter_used"] = int(used_phase_filter and measured_only)
        if stripes:
            stripe_durs = _stats([s.dur for s in stripes])
            derived["per_qp_stripe_dur_avg_us"] = round(stripe_durs["avg"], 3)
            derived["per_qp_stripe_dur_max_us"] = round(stripe_durs["max"], 3)
        if logical:
            logical_durs = _stats([item["duration_us"] for item in logical])
            logical_bws = _stats([item["bw_gb_s"] for item in logical])
            derived["logical_dur_avg_us"] = round(logical_durs["avg"], 3)
            derived["logical_dur_max_us"] = round(logical_durs["max"], 3)
            derived["logical_bw_avg_gb_s"] = round(logical_bws["avg"], 3)
            derived["logical_bw_max_gb_s"] = round(logical_bws["max"], 3)
            derived["start_skew_avg_us"] = round(
                _stats([item["start_skew_us"] for item in logical])["avg"], 3
            )
            derived["completion_skew_avg_us"] = round(
                _stats([item["completion_skew_us"] for item in logical])["avg"], 3
            )

        if stripes_csv:
            write_detail_csv(
                stripes_csv,
                [
                    {
                        "row_type": "per-QP stripe",
                        "source": os.path.basename(s.source),
                        "pid": s.pid,
                        "id": s.xid,
                        "qp": s.qp,
                        "key": s.key,
                        "op": s.op,
                        "bytes": s.bytes,
                        "wrs": s.wrs,
                        "merged": s.merged,
                        "post_us": f"{s.post:.3f}",
                        "cqe_us": f"{s.cqe:.3f}",
                        "duration_us": f"{s.dur:.3f}",
                        "stripe_bw_gb_s": f"{s.bw:.6f}",
                    }
                    for s in stripes
                ],
                [
                    "row_type", "source", "pid", "id", "qp", "key", "op", "bytes",
                    "wrs", "merged", "post_us", "cqe_us", "duration_us", "stripe_bw_gb_s",
                ],
            )
            print(f"  [saved] per-QP stripe CSV -> {stripes_csv}")
        if logical_csv:
            write_detail_csv(
                logical_csv,
                [
                    {
                        "row_type": "logical transfer",
                        "source": os.path.basename(item["source"]),
                        "pid": item["pid"],
                        "id": item["id"],
                        "op": item["op"],
                        "qps": ";".join(str(qp) for qp in item["qps"]),
                        "stripe_count": len(item["stripes"]),
                        "bytes": item["bytes"],
                        "duration_us": f"{item['duration_us']:.3f}",
                        "logical_bw_gb_s": f"{item['bw_gb_s']:.6f}",
                        "start_skew_us": f"{item['start_skew_us']:.3f}",
                        "completion_skew_us": f"{item['completion_skew_us']:.3f}",
                    }
                    for item in logical
                ],
                [
                    "row_type", "source", "pid", "id", "op", "qps", "stripe_count",
                    "bytes", "duration_us", "logical_bw_gb_s", "start_skew_us",
                    "completion_skew_us",
                ],
            )
            print(f"  [saved] logical transfer CSV -> {logical_csv}")
    elif kv:
        print(
            "\n[!] DEGRADED RANGE-ONLY INFERENCE: no qp_post/qp_cqe instant marks. "
            "Legacy async ranges may be corrupted by cross-thread ROCTx correlation "
            "and are not authoritative per-QP timings."
        )

    # ---- Retained range visualization / backward compatibility --------------
    if kv:
        print("\n[legacy] ASYNC RANGE VISUALIZATION (not keyed timing authority)")
        qp_values = sorted({r.qp for r in kv if r.qp is not None})
        if qp_values:
            print(f"  observed QP indices: {qp_values}")
            derived["qp_count"] = len(qp_values)
            derived["qp_indices"] = ";".join(str(qp) for qp in qp_values)
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
        if line_rate_gbps > 0 and "logical_bw_max_gb_s" in derived:
            derived["logical_eff_vs_line_pct"] = round(
                100.0 * (derived["logical_bw_max_gb_s"] * 8.0) / line_rate_gbps, 2
            )
        elif line_rate_gbps > 0 and "per_transfer_bw_max_gbps" in derived:
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
    p.add_argument(
        "--out-stripes-csv",
        default="",
        help="Write one correctly joined 'per-QP stripe' row per submission key.",
    )
    p.add_argument(
        "--out-logical-csv",
        default="",
        help="Write logical-transfer rows grouped across all QP stripes.",
    )
    p.add_argument(
        "--include-unmeasured",
        action="store_true",
        help="Include setup/warmup events even when explicit measured-phase marks exist.",
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

    analyze(
        all_ranges,
        out_csv=args.out_csv or None,
        line_rate_gbps=args.line_rate_gbps,
        stripes_csv=args.out_stripes_csv or None,
        logical_csv=args.out_logical_csv or None,
        measured_only=not args.include_unmeasured,
    )


if __name__ == "__main__":
    main()
