from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class SegmentRange:
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def segment_sequence(num_frames: int, segment_length: int, overlap: int) -> List[SegmentRange]:
    """Split a sequence into overlapping hard-reset segments."""
    if num_frames <= 0:
        return []
    if segment_length <= 0:
        raise ValueError("segment_length must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= segment_length:
        raise ValueError("overlap must be smaller than segment_length")

    step = segment_length - overlap
    ranges: List[SegmentRange] = []
    start = 0
    while start < num_frames:
        end = min(start + segment_length, num_frames)
        ranges.append(SegmentRange(start=start, end=end))
        if end >= num_frames:
            break
        start += step
    return ranges
