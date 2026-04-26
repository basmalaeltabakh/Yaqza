"""Windowing helpers (placeholders)."""
from typing import Iterable, List


def sliding_window(sequence: Iterable[float], window_size: int) -> List[List[float]]:
    """Return a list of sliding windows of given size from sequence."""
    seq = list(sequence)
    if window_size <= 0:
        return []
    return [seq[i:i+window_size] for i in range(len(seq) - window_size + 1)]
