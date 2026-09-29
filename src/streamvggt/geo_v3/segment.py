from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class SegmentRange:
    """一个分段在整段序列中的帧区间，半开区间 ``[start, end)``。"""
    start: int
    end: int

    @property
    def length(self) -> int:
        """该分段包含的帧数 = end - start。"""
        return self.end - self.start


def segment_sequence(num_frames: int, segment_length: int, overlap: int) -> List[SegmentRange]:
    """把一段长度为 ``num_frames`` 的序列切成若干带重叠的 hard-reset 分段。

    相邻分段之间共享 ``overlap`` 帧，用于跨段的重叠传播 / 位姿对齐。
    步长 step = segment_length - overlap，即每段相对上一段前进 step 帧。

    参数：
        num_frames:     序列总帧数；<= 0 时返回空列表。
        segment_length: 单段长度（帧数），必须为正。
        overlap:        相邻段重叠帧数，需满足 0 <= overlap < segment_length。

    返回：
        覆盖 ``[0, num_frames)`` 的 SegmentRange 列表；最后一段在抵达序列末尾时收尾，
        长度可能小于 segment_length。

    异常：
        segment_length <= 0、overlap < 0 或 overlap >= segment_length 时抛 ValueError。
    """
    if num_frames <= 0:
        return []
    if segment_length <= 0:
        raise ValueError("segment_length must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= segment_length:
        # 重叠不能 >= 段长，否则步长 <= 0 会导致无法前进 / 死循环。
        raise ValueError("overlap must be smaller than segment_length")

    step = segment_length - overlap  # 每段相对上一段前进的帧数
    ranges: List[SegmentRange] = []
    start = 0
    while start < num_frames:
        # 段尾不超过序列末尾（最后一段可能不足 segment_length）。
        end = min(start + segment_length, num_frames)
        ranges.append(SegmentRange(start=start, end=end))
        if end >= num_frames:
            # 已覆盖到末尾，收尾退出（避免再生成一段重复尾部的零长 / 越界段）。
            break
        start += step
    return ranges
