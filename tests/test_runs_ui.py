from yylab.ui.pages.runs import (
    EMPTY_POLL_SECONDS,
    FAST_POLL_SECONDS,
    IDLE_POLL_SECONDS,
    LOG_TAIL_BYTES,
    _poll_interval,
    _read_log_chunk,
)


def test_runs_poll_interval_tracks_activity_and_selection():
    assert _poll_interval(has_active_runs=True, has_selection=False) == FAST_POLL_SECONDS
    assert _poll_interval(has_active_runs=False, has_selection=True) == IDLE_POLL_SECONDS
    assert _poll_interval(has_active_runs=False, has_selection=False) == EMPTY_POLL_SECONDS


def test_log_chunk_reads_tail_then_only_appended_bytes(tmp_path):
    log = tmp_path / "train.log"
    log.write_bytes(b"loaded\nstep 1\n")

    text, offset, identity, mtime_ns, reset = _read_log_chunk(log, offset=None, identity=None, mtime_ns=None)
    assert reset
    assert text == "loaded\nstep 1\n"

    with log.open("ab") as stream:
        stream.write(b"step 2\n")
    text, offset, identity, mtime_ns, reset = _read_log_chunk(
        log, offset=offset, identity=identity, mtime_ns=mtime_ns
    )
    assert not reset
    assert text == "step 2\n"

    text, *_rest, reset = _read_log_chunk(log, offset=offset, identity=identity, mtime_ns=mtime_ns)
    assert not reset
    assert text == ""


def test_log_chunk_resets_after_truncation_and_bounds_initial_tail(tmp_path):
    log = tmp_path / "train.log"
    log.write_bytes(b"line\n" * (LOG_TAIL_BYTES // 2))

    text, offset, identity, mtime_ns, reset = _read_log_chunk(log, offset=None, identity=None, mtime_ns=None)
    assert reset
    assert len(text.encode()) <= LOG_TAIL_BYTES
    assert text.startswith("line\n")

    log.write_bytes(b"x\n")
    text, _offset, _identity, _mtime_ns, reset = _read_log_chunk(
        log, offset=offset, identity=identity, mtime_ns=mtime_ns
    )
    assert reset
    assert text == "x\n"
