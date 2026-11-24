"""Invalidation of the cached Numba kernels when jitted code changes."""

from twpasolver import _drop_stale_numba_cache


def test_cache_dropped_only_when_jitted_source_changes(tmp_path):
    src = tmp_path / "kern.py"
    src.write_text("@njit(cache=True)\ndef f(): return 1\n")
    (tmp_path / "plain.py").write_text("x = 1\n")
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    entry = cache / "kern.f-1.py310.nbi"

    entry.touch()
    _drop_stale_numba_cache(tmp_path)  # no stamp yet: drop
    assert not entry.exists()

    entry.touch()
    (tmp_path / "plain.py").write_text("x = 2\n")  # not jitted: keep
    _drop_stale_numba_cache(tmp_path)
    assert entry.exists()

    src.write_text("@njit(cache=True)\ndef f(): return 2\n")  # jitted edit: drop
    _drop_stale_numba_cache(tmp_path)
    assert not entry.exists()
