"""公开二进制开关交换格式。

文件布局：4 字节小端目录长度 + 目录 JSON + 条目数据。目录报告层、
数据版、通道、开关网格、块划分，以及每个块的坐标范围、文件偏移与
条目数；并附层几何，解码端可依据公开开关格推回潜在影响范围。

条目为小端 ``<HHhhB``：开关格 y、x，源 y、x 和标记
（0 无覆盖、1 确定、2 未确定）；未确定项源坐标写 −1。源坐标
(0, 0) 配合标记 1 是有效零，与无覆盖、待确认三者分得清。

只传池化值不够：本文件只携带开关，池化值经张量路径传递
（``decode_region`` 的 ``pooled_values``）。
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

from .engine import (
    Budget,
    CellResult,
    FLAG_DETERMINED,
    FLAG_NO_COVERAGE,
    SwitchIssue,
)
from .geometry import LayerSpec, footprint

ENTRY = struct.Struct("<HHhhB")
HEADER = struct.Struct("<I")
FORMAT = "poolswitch-switches/1"
DEFAULT_BLOCK = (16, 16)


def _spec_to_json(spec: LayerSpec) -> dict:
    return {
        "shape": list(spec.shape),
        "kernel": list(spec.kernel),
        "stride": list(spec.stride),
        "dilation": list(spec.dilation),
        "padding": list(spec.padding),
        "grid": list(spec.grid),
        "origin": list(spec.origin),
        "channels": spec.channels,
    }


def _spec_from_json(name: str, raw: dict) -> LayerSpec:
    return LayerSpec(
        name=name,
        shape=tuple(raw["shape"]),
        kernel=tuple(raw["kernel"]),
        stride=tuple(raw["stride"]),
        dilation=tuple(raw["dilation"]),
        padding=tuple(raw["padding"]),
        grid=tuple(raw["grid"]),
        origin=tuple(raw["origin"]),
        channels=int(raw.get("channels", 1)),
    )


def write_switch_file(path, issue: SwitchIssue, channel: int = 0, block=DEFAULT_BLOCK) -> None:
    """把一次出具的开关快照写成公开二进制文件（每层每通道一个文件）。"""
    spec = issue.spec
    gh, gw = spec.grid
    bh, bw = block
    if gh > 0xFFFF or gw > 0xFFFF:
        raise ValueError("开关网格超出 <H 可表示范围")
    payload = bytearray()
    blocks = []
    for by in range(0, gh, bh):
        for bx in range(0, gw, bw):
            y1, x1 = min(by + bh, gh), min(bx + bw, gw)
            start = len(payload)
            count = 0
            for gy in range(by, y1):
                for gx in range(bx, x1):
                    flag = int(issue.flags[gy, gx, channel])
                    if flag == FLAG_DETERMINED:
                        sy, sx = int(issue.src_y[gy, gx, channel]), int(issue.src_x[gy, gx, channel])
                        if not (0 <= sy <= 0x7FFF and 0 <= sx <= 0x7FFF):
                            raise ValueError("源坐标超出 <h 可表示范围")
                    else:
                        sy = sx = -1
                    payload += ENTRY.pack(gy, gx, sy, sx, flag)
                    count += 1
            blocks.append({"y0": by, "y1": y1, "x0": bx, "x1": x1, "offset": start, "count": count})
    directory = {
        "format": FORMAT,
        "layer": issue.layer,
        "data_version": issue.data_version,
        "channel": int(channel),
        "layer_spec": _spec_to_json(spec),
        "grid": [gh, gw],
        "block": [bh, bw],
        "blocks": blocks,
    }
    # 偏移从数据区起点换算为文件偏移；目录长度随偏移位数变化，迭代到不动点
    relative = [b["offset"] for b in blocks]
    offsets = list(relative)
    directory_json = b""
    for _ in range(10):
        for b, off in zip(blocks, offsets):
            b["offset"] = off
        directory_json = json.dumps(directory, ensure_ascii=False, sort_keys=True).encode("utf-8")
        base = HEADER.size + len(directory_json)
        new_offsets = [base + r for r in relative]
        if new_offsets == offsets:
            break
        offsets = new_offsets
    else:
        raise RuntimeError("目录偏移未收敛")
    Path(path).write_bytes(HEADER.pack(len(directory_json)) + directory_json + bytes(payload))


class _FileProvider:
    """二进制开关文件作为单元提供者（边端解码侧）。"""

    def __init__(self, switch_file: "SwitchFile"):
        self.file = switch_file
        self.spec = switch_file.spec
        self.issue_id = ("file", switch_file.layer, switch_file.data_version, switch_file.channel)

    def cell(self, gy: int, gx: int, c: int = 0) -> CellResult:
        return self.file.cell(gy, gx, c)


class SwitchFile:
    """按块目录做范围读取：非整图查询不必先解出全部开关。"""

    def __init__(self, path, budget: Budget | None = None):
        raw = Path(path).read_bytes()
        (length,) = HEADER.unpack(raw[: HEADER.size])
        directory = json.loads(raw[HEADER.size : HEADER.size + length].decode("utf-8"))
        if directory.get("format") != FORMAT:
            raise ValueError(f"未知开关文件格式 {directory.get('format')!r}")
        self.directory = directory
        self.layer = directory["layer"]
        self.data_version = int(directory["data_version"])
        self.channel = int(directory.get("channel", 0))
        self.spec = _spec_from_json(self.layer, directory["layer_spec"])
        self._raw = raw
        self._blocks = directory["blocks"]
        self._block_shape = tuple(directory["block"])
        self._blocks_per_row = (self.spec.grid[1] + self._block_shape[1] - 1) // self._block_shape[1]
        self._cache: dict[tuple[int, int], dict[tuple[int, int], tuple[int, int, int]]] = {}
        self.blocks_read = 0  # 诊断：实际读了多少块
        self.budget = budget

    # ---- 块级读取 ----------------------------------------------------

    def _block_info(self, gy: int, gx: int) -> dict:
        bh, bw = self._block_shape
        return self._blocks[(gy // bh) * self._blocks_per_row + (gx // bw)]

    def _read_block(self, info: dict) -> dict[tuple[int, int], tuple[int, int, int]]:
        key = (info["y0"], info["x0"])
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        offset, count = info["offset"], info["count"]
        if self.budget is not None:
            self.budget.charge(Budget.SEARCH, count * ENTRY.size)
            self.budget.charge(Budget.STAGING, count * ENTRY.size)
        entries = {}
        for i in range(count):
            gy, gx, sy, sx, flag = ENTRY.unpack_from(self._raw, offset + i * ENTRY.size)
            entries[(gy, gx)] = (flag, sy, sx)
        self.blocks_read += 1
        self._cache[key] = entries
        return entries

    # ---- 公开查询 ----------------------------------------------------

    def query(self, gy0: int, gy1: int, gx0: int, gx1: int) -> list[tuple[int, int, int, int, int]]:
        """读取开关格子范围 [gy0,gy1)×[gx0,gx1)，只解相交的块。"""
        gh, gw = self.spec.grid
        gy0, gy1 = max(gy0, 0), min(gy1, gh)
        gx0, gx1 = max(gx0, 0), min(gx1, gw)
        out = []
        if gy0 >= gy1 or gx0 >= gx1:
            return out
        bh, bw = self._block_shape
        for by in range(gy0 // bh, (gy1 - 1) // bh + 1):
            for bx in range(gx0 // bw, (gx1 - 1) // bw + 1):
                entries = self._read_block(self._blocks[by * self._blocks_per_row + bx])
                for (gy, gx), (flag, sy, sx) in entries.items():
                    if gy0 <= gy < gy1 and gx0 <= gx < gx1:
                        out.append((gy, gx, flag, sy, sx))
        out.sort()
        return out

    def cell(self, gy: int, gx: int, c: int = 0) -> CellResult:
        if c != self.channel:
            raise ValueError(f"本文件只携带通道 {self.channel}")
        flag, sy, sx = self._read_block(self._block_info(gy, gx))[(gy, gx)]
        if flag == FLAG_DETERMINED:
            return CellResult(flag, None, sy, sx, c)  # 池化值经张量路径传递
        if flag == FLAG_NO_COVERAGE:
            return CellResult(flag, float("-inf"), None, None, c)
        return CellResult(flag, None, None, None, c)

    def potential_influence(self, gy: int, gx: int):
        """依据公开开关格推回的潜在影响范围（原图坐标，左闭右开）。"""
        fp = footprint(self.spec, gy, gx)
        if fp is None:
            return None
        oy, ox = self.spec.origin
        return (fp[0] + oy, fp[1] + oy, fp[2] + ox, fp[3] + ox)

    def as_provider(self) -> _FileProvider:
        return _FileProvider(self)
