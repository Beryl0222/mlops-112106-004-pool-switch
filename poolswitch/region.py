"""区域解码与跨层回填：边端与训练共用的张量路径。

同一条路径接受三种单元提供者（CellProvider 协议：``spec`` 属性与
``cell(gy, gx, c)`` 方法）——训练侧的活引擎、交接用的开关快照、
边端的二进制开关文件。

融合约定：同一原图位置若被多层的窗口命中，贡献按层号、开关格坐标、
通道的顺序以 FP64 累加。区域完整性与浮点输出分开报告：只要某个
未确定开关的潜在影响范围（窗口 ∩ 层图像，按 origin 平移）与查询
区域相交，该区域就不能冒充完整结果——即便目前只有别的区域报缺片。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .engine import (
    Budget,
    CellResult,
    FLAG_DETERMINED,
    FLAG_UNDETERMINED,
    normalize_dy,
)
from .geometry import cells_intersecting


def layer_sort_key(name: str):
    """层号的自然序：l2 排在 l10 前。"""
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in re.split(r"(\d+)", name)
    )


@dataclass
class RegionReport:
    """一次区域查询的结果：浮点输出与受影响区域的完整性分开。"""

    rect: tuple[int, int, int, int]
    values: np.ndarray  # (h, w) 或 (h, w, C)，FP64
    complete: bool  # 没有未确定开关可能命中该区域
    pending: list = field(default_factory=list)  # (layer, gy, gx, c) 未确定的开关


class GradientRegistry:
    """按开关格登记预期贡献：训练梯度以实际出具的开关版为准。"""

    def __init__(self):
        self._entries: dict[tuple, dict[tuple[int, int, int], float]] = {}

    def record(self, layer: str, issue_id, gy: int, gx: int, c: int, value: float) -> None:
        self._entries.setdefault((layer, issue_id), {})[(gy, gx, c)] = float(value)

    def expected(self, layer: str | None = None, issue_id=None) -> dict:
        if layer is None:
            return {key: dict(value) for key, value in self._entries.items()}
        return dict(self._entries.get((layer, issue_id), {}))


def _ordered_layers(providers) -> list[str]:
    return sorted(providers, key=layer_sort_key)


def _sub_rect(rect, spec) -> tuple[int, int, int, int]:
    """查询区域（原图坐标）平移并裁剪到层坐标。"""
    y0, y1, x0, x1 = rect
    oy, ox = spec.origin
    return (
        max(y0 - oy, 0),
        min(y1 - oy, spec.height),
        max(x0 - ox, 0),
        min(x1 - ox, spec.width),
    )


def _pooled_value(provider, pooled_values, lname, gy, gx, c, result: CellResult) -> float:
    """开关源不带池化值时（如二进制文件），从张量路径取。"""
    if result.value is not None:
        return result.value
    if pooled_values is None or lname not in pooled_values:
        raise ValueError(f"开关源 {lname} 不带池化值，需要 pooled_values 补足")
    arr = np.asarray(pooled_values[lname], dtype=np.float64)
    if arr.ndim == 2:
        return float(arr[gy, gx])
    return float(arr[gy, gx, c])


def decode_region(rect, providers, pooled_values=None, budget: Budget | None = None) -> RegionReport:
    """把若干池化层的激活按胜出位置回填到原图查询区域。"""
    y0, y1, x0, x1 = (int(v) for v in rect)
    channels = max(p.spec.channels for p in providers.values())
    out = np.zeros((y1 - y0, x1 - x0, channels), dtype=np.float64)
    pending: list = []
    for lname in _ordered_layers(providers):
        provider = providers[lname]
        spec = provider.spec
        sub = _sub_rect(rect, spec)
        if sub[0] >= sub[1] or sub[2] >= sub[3]:
            continue
        oy, ox = spec.origin
        for gy, gx in cells_intersecting(spec, sub):
            if budget is not None:
                budget.charge(Budget.SEARCH, Budget.CELL_SEARCH_BYTES)
            for c in range(spec.channels):
                r = provider.cell(gy, gx, c)
                if r.flag == FLAG_UNDETERMINED:
                    # 窗口与本区域相交 ⇒ 潜在影响范围命中本区域
                    pending.append((lname, gy, gx, c))
                    continue
                if r.flag != FLAG_DETERMINED:
                    continue  # 无覆盖：没有可回填的激活
                Y, X = r.y + oy, r.x + ox
                if not (y0 <= Y < y1 and x0 <= X < x1):
                    continue
                value = _pooled_value(provider, pooled_values, lname, gy, gx, c, r)
                if budget is not None:
                    budget.charge(Budget.STAGING, Budget.STAGING_SLOT_BYTES)
                out[Y - y0, X - x0, r.c] += value
    values = out[:, :, 0] if channels == 1 else out
    return RegionReport((y0, y1, x0, x1), values, not pending, pending)


def backward_region(rect, providers, pool_dy, registry: GradientRegistry | None = None,
                    budget: Budget | None = None) -> RegionReport:
    """把各有效窗口的上游梯度按胜出位置送回传感器特征（原图坐标）。"""
    y0, y1, x0, x1 = (int(v) for v in rect)
    channels = max(p.spec.channels for p in providers.values())
    out = np.zeros((y1 - y0, x1 - x0, channels), dtype=np.float64)
    pending: list = []
    for lname in _ordered_layers(providers):
        provider = providers[lname]
        spec = provider.spec
        if lname not in pool_dy:
            raise ValueError(f"层 {lname} 缺少 pool_dy")
        dy_values, dy_arrived = normalize_dy(pool_dy[lname], spec)
        sub = _sub_rect(rect, spec)
        if sub[0] >= sub[1] or sub[2] >= sub[3]:
            continue
        oy, ox = spec.origin
        for gy, gx in cells_intersecting(spec, sub):
            if budget is not None:
                budget.charge(Budget.SEARCH, Budget.CELL_SEARCH_BYTES)
            for c in range(spec.channels):
                r = provider.cell(gy, gx, c)
                if r.flag == FLAG_UNDETERMINED:
                    pending.append((lname, gy, gx, c))
                    continue
                if r.flag != FLAG_DETERMINED:
                    continue
                Y, X = r.y + oy, r.x + ox
                if not (y0 <= Y < y1 and x0 <= X < x1):
                    continue
                if not dy_arrived[gy, gx]:
                    raise ValueError(f"层 {lname} 开关格 {(gy, gx)} 的反向贡献未送齐")
                contribution = float(dy_values[gy, gx, c])
                if budget is not None:
                    budget.charge(Budget.STAGING, Budget.STAGING_SLOT_BYTES)
                out[Y - y0, X - x0, r.c] += contribution
                if registry is not None:
                    registry.record(lname, provider.issue_id, gy, gx, c, contribution)
    values = out[:, :, 0] if channels == 1 else out
    return RegionReport((y0, y1, x0, x1), values, not pending, pending)
