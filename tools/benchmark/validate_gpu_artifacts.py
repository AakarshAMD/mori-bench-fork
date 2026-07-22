#!/usr/bin/env python3
"""Fail unless a GPU benchmark run contains the required data and trace evidence."""

import csv
import glob
import os
import sys


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def data_rows(path):
    with open(path, newline="", encoding="utf-8", errors="replace") as stream:
        return max(0, sum(1 for _ in csv.reader(stream)) - 1)


def main():
    root = os.path.abspath(sys.argv[1])

    for phase in ("4a", "4b", "p1", "p2"):
        for kind in ("summary", "raw"):
            path = os.path.join(root, "out", f"{phase}_{kind}.csv")
            require(os.path.isfile(path), f"missing {path}")
            rows = data_rows(path)
            require(rows > 0, f"no data rows in {path}")
            print(f"CSV_ROWS {phase}_{kind}={rows}")

    logs = sorted(glob.glob(os.path.join(root, "logs", "rank*_port*.log")))
    require(len(logs) == 8, f"expected 8 rank logs, found {len(logs)}")
    for path in logs:
        text = open(path, encoding="utf-8", errors="replace").read()
        require("mem_type: gpu" in text, f"GPU memory mode missing in {path}")
        require("MORI_ENABLE_DMABUF_REG=1" in text, f"DMABUF evidence missing in {path}")
        require("MORI_CPP_FILE=" in text and "MORI_NATIVE_SOS=" in text, f"native import evidence missing in {path}")
    print(f"GPU_LOGS mem_type_gpu={len(logs)} dmabuf_enabled={len(logs)}")

    for phase in ("p1", "p2"):
        trace_dir = os.path.join(root, f"rocprof_{phase}")
        markers = sorted(glob.glob(os.path.join(trace_dir, "*_marker_api_trace.csv")))
        jsons = sorted(glob.glob(os.path.join(trace_dir, "*.json")))
        pftraces = sorted(glob.glob(os.path.join(trace_dir, "*.pftrace")))
        require(markers, f"no marker CSV in {trace_dir}")
        require(jsons, f"no JSON trace in {trace_dir}")
        require(pftraces, f"no pftrace in {trace_dir}")
        qp_rows = 0
        for path in markers:
            with open(path, encoding="utf-8", errors="replace") as stream:
                qp_rows += sum("mori.rdma.kv_transfer" in line and "qp=" in line for line in stream)
        require(qp_rows > 0, f"no kv_transfer qp= marker rows in {trace_dir}")
        print(
            f"TRACE {phase} marker_csv={len(markers)} qp_rows={qp_rows} "
            f"json={len(jsons)} pftrace={len(pftraces)}"
        )


if __name__ == "__main__":
    main()
