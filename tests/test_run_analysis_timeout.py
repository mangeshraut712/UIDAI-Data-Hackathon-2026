import sys
import time

from run_analysis import stream_process


def test_stream_process_kills_a_child_that_outlives_the_timeout():
    lines = []
    start = time.time()
    returncode, timed_out = stream_process(
        [sys.executable, "-c", "print('started', flush=True); import time; time.sleep(30)"],
        timeout=0.5,
        emit=lines.append,
    )
    assert timed_out
    assert returncode != 0
    assert time.time() - start < 10
    assert lines == ["  started"]


def test_stream_process_reports_a_normal_exit():
    lines = []
    returncode, timed_out = stream_process(
        [sys.executable, "-c", "print('a'); print('b')"], timeout=30, emit=lines.append
    )
    assert (returncode, timed_out) == (0, False)
    assert lines == ["  a", "  b"]
