import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest


TOOL = Path(__file__).parents[3] / "tools" / "profiler" / "analyze_io_marker_trace.py"
SPEC = importlib.util.spec_from_file_location("analyze_io_marker_trace", TOOL)
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)


def event(kind, key, xid, qp, timestamp, source="rank0.csv", pid="100", byte_count=1024):
    name = (
        f"mori.rdma.{kind} key={key} id={xid} qp={qp} bytes={byte_count} "
        f"wrs=1 merged=1 op=write"
    )
    return ANALYZER.Range(name, timestamp, timestamp, "7", pid, source)


class KeyedQpTimingTest(unittest.TestCase):
    def test_csv_parses_keyed_instant_marks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rank0_marker_api_trace.csv"
            path.write_text(
                '"Domain","Function","Process_Id","Thread_Id","Correlation_Id",'
                '"Start_Timestamp","End_Timestamp"\n'
                '"MARKER_CORE_MARK_API","mori.rdma.qp_post key=9 id=3 qp=2 '
                'bytes=4096 wrs=1 merged=1 op=write",10,11,1,1000,1000\n'
                '"MARKER_CORE_MARK_API","mori.rdma.qp_cqe key=9 id=3 qp=2 '
                'bytes=4096 wrs=1 merged=1 op=write",10,12,2,5000,5000\n'
            )
            records = ANALYZER.parse_marker_csv(str(path), "ns")
            stripes, issues, _ = ANALYZER.join_qp_events(records)
            self.assertEqual(issues, [])
            self.assertEqual(len(stripes), 1)
            self.assertEqual(stripes[0].dur, 4.0)
            self.assertEqual(stripes[0].bytes, 4096)

    def test_four_qps_interleaved_completion_order(self):
        records = [event("qp_post", qp + 1, 42, qp, 10 + qp) for qp in range(4)]
        records += [
            event("qp_cqe", 4, 42, 3, 24),
            event("qp_cqe", 2, 42, 1, 25),
            event("qp_cqe", 1, 42, 0, 26),
            event("qp_cqe", 3, 42, 2, 27),
        ]
        stripes, issues, _ = ANALYZER.join_qp_events(records)
        self.assertEqual(issues, [])
        self.assertEqual(len(stripes), 4)
        logical = ANALYZER.group_logical_transfers(stripes)
        self.assertEqual(len(logical), 1)
        self.assertEqual(logical[0]["qps"], [0, 1, 2, 3])
        self.assertEqual(logical[0]["bytes"], 4096)
        self.assertEqual(logical[0]["start_skew_us"], 3)
        self.assertEqual(logical[0]["completion_skew_us"], 3)

    def test_duplicate_and_missing_endpoints_are_reported(self):
        post = event("qp_post", 1, 5, 0, 10)
        records = [post, post, event("qp_cqe", 2, 5, 1, 20)]
        stripes, issues, _ = ANALYZER.join_qp_events(records)
        self.assertEqual(stripes, [])
        self.assertEqual(len(issues), 2)
        self.assertTrue(any("post=2 cqe=0" in issue for issue in issues))
        self.assertTrue(any("post=0 cqe=1" in issue for issue in issues))

    def test_same_ids_and_keys_are_isolated_by_trace_and_process(self):
        records = []
        processes = (
            ("rank0.csv", "10", 1),
            ("rank1.csv", "10", 5),
            ("rank0.csv", "11", 9),
        )
        for source, pid, start in processes:
            records.extend(
                [
                    event("qp_post", 1, 0, 0, start, source, pid),
                    event("qp_cqe", 1, 0, 0, start + 2, source, pid),
                ]
            )
        stripes, issues, _ = ANALYZER.join_qp_events(records)
        self.assertEqual(issues, [])
        self.assertEqual(len(stripes), 3)
        self.assertEqual(len(ANALYZER.group_logical_transfers(stripes)), 3)

    def test_measured_phase_excludes_warmup(self):
        phase_begin = ANALYZER.Range(
            "mori.bench.phase measured_begin", 100, 100, "1", "10", "rank0.csv"
        )
        phase_end = ANALYZER.Range(
            "mori.bench.phase measured_end", 200, 200, "1", "10", "rank0.csv"
        )
        records = [
            event("qp_post", 1, 0, 0, 10, pid="10"),
            event("qp_cqe", 1, 0, 0, 20, pid="10"),
            phase_begin,
            event("qp_post", 2, 1, 0, 120, pid="10"),
            event("qp_cqe", 2, 1, 0, 140, pid="10"),
            phase_end,
        ]
        stripes, issues, used_phase = ANALYZER.join_qp_events(records)
        self.assertEqual(issues, [])
        self.assertTrue(used_phase)
        self.assertEqual([stripe.xid for stripe in stripes], [1])

    def test_range_only_csv_remains_supported_and_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old_marker_api_trace.csv"
            path.write_text(
                '"Domain","Function","Process_Id","Thread_Id","Correlation_Id",'
                '"Start_Timestamp","End_Timestamp"\n'
                '"MARKER_CORE_RANGE_API","mori.rdma.kv_transfer bytes=1024 wrs=1 '
                'merged=1 qp=0 id=3",10,11,1,1000,2000\n'
            )
            records = ANALYZER.parse_marker_csv(str(path), "ns")
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].dur, 1.0)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                ANALYZER.analyze(records)
            self.assertIn("DEGRADED RANGE-ONLY INFERENCE", output.getvalue())


if __name__ == "__main__":
    unittest.main()
