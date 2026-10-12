"""Concurrent result accumulation on a local POSIX filesystem."""

from __future__ import annotations

import importlib.util
import json
import multiprocessing
import os
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parents[1] / "experiments/protocol_swap/source/runner.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("protocol_concurrency", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_worker(output, model, role, snapshot_ready, release_first, lock_attempted, outcomes):
    import fcntl

    module = load_runner()
    if role == "first":
        original_replace = module.os.replace

        def paused_replace(source, destination):
            # Pause after serialization but before commit, with the transaction open.
            snapshot_ready.set()
            if not release_first.wait(15):
                raise TimeoutError("first writer was not released")
            return original_replace(source, destination)

        module.os.replace = paused_replace
    else:
        original_flock = fcntl.flock

        def observed_flock(fd, operation):
            if operation == fcntl.LOCK_EX:
                lock_attempted.set()
            return original_flock(fd, operation)

        fcntl.flock = observed_flock
    try:
        module.save_results_atomically(Path(output), model, {"writer": role})
        outcomes.put((role, "saved"))
    except Exception as error:
        outcomes.put((role, str(error)))


@pytest.mark.skipif(os.name != "posix", reason="protocol runner requires POSIX file locking")
@pytest.mark.parametrize("duplicate", [False, True])
def test_concurrent_model_saves_keep_all_entries(tmp_path, duplicate):
    ctx = multiprocessing.get_context("spawn")
    ready, release, attempting = ctx.Event(), ctx.Event(), ctx.Event()
    outcomes = ctx.Queue()
    output = tmp_path.resolve() / "results.json"
    first = ctx.Process(target=save_worker, args=(
        str(output), "model-a", "first", ready, release, attempting, outcomes,
    ))
    second = ctx.Process(target=save_worker, args=(
        str(output), "model-a" if duplicate else "model-b", "second",
        ready, release, attempting, outcomes,
    ))
    first.start()
    try:
        assert ready.wait(15), "first writer did not reach its pending commit"
        second.start()
        assert attempting.wait(15), "second writer did not attempt the transaction lock"
        release.set()
        first.join(15)
        second.join(15)
        assert first.exitcode == second.exitcode == 0
        statuses = dict(outcomes.get(timeout=5) for _ in range(2))
        assert statuses["first"] == "saved"
        expected = {"model-a": {"writer": "first"}}
        if duplicate:
            assert "already contains results" in statuses["second"]
        else:
            assert statuses["second"] == "saved"
            expected["model-b"] = {"writer": "second"}
        assert json.loads(output.read_text()) == expected
        assert sorted(p.name for p in tmp_path.iterdir()) == [".results.json.lock", "results.json"]
    finally:
        release.set()
        for process in (first, second):
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(5)
        outcomes.close()
        outcomes.join_thread()


@pytest.mark.skipif(os.name != "posix", reason="protocol runner requires POSIX file locking")
def test_lock_symlink_is_rejected_without_touching_target(tmp_path):
    module = load_runner()
    output = tmp_path.resolve() / "results.json"
    target = tmp_path / "unrelated.txt"
    target.write_text("unchanged")
    output.with_name(".results.json.lock").symlink_to(target)
    with pytest.raises(OSError):
        module.save_results_atomically(output, "model-a", {})
    assert target.read_text() == "unchanged"
    assert not output.exists()
