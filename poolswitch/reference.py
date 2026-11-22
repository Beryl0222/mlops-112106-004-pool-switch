"""全图稠密参照：对照整层定义检验流式引擎与共用路径。

这里不做任何流式优化，直接把全部主记录拼成整层稠密张量后逐格
求开关，用于在换瓦片划分、换请求区域之后，检验同一激活和梯度
落在何处。
"""

from __future__ import annotations

import numpy as np

from .engine import (
    CellResult,
    Engine,
    FLAG_DETERMINED,
    FLAG_NO_COVERAGE,
    normalize_dy,
)
from .geometry import in_bounds, window_positions
from .region import layer_sort_key
from .scene import Scene


def dense_arrays(scene: Scene, layer: str):
    """把全部主记录拼成整层稠密张量：值 (H,W,C)、已观测 (H,W)、可拥有 (H,W)。"""
    spec = scene.layers[layer]
    values = np.full((spec.height, spec.width, spec.channels), np.nan)
    observed = np.zeros((spec.height, spec.width), dtype=bool)
    ownable = np.zeros((spec.height, spec.width), dtype=bool)
    for owner, rect in scene.primary[layer].items():
        y0, y1, x0, x1 = rect
        ownable[y0:y1, x0:x1] = True
        tile = scene.tiles[layer][owner]
        values[y0:y1, x0:x1] = tile.values
        observed[y0:y1, x0:x1] = tile.observed
    return values, observed, ownable


def reference_cells(spec, values, observed, ownable) -> dict[tuple[int, int, int], CellResult]:
    """按整层定义逐格求开关：越界负无穷、未观测排除、平局按 (y, x, 通道)。"""
    gh, gw = spec.grid
    out: dict[tuple[int, int, int], CellResult] = {}
    for gy in range(gh):
        for gx in range(gw):
            for c in range(spec.channels):
                best = None
                for y, x in window_positions(spec, gy, gx):
                    if not in_bounds(spec, y, x):
                        continue
                    if not (ownable[y, x] and observed[y, x]):
                        continue
                    v = float(values[y, x, c])
                    if best is None or v > best[0] or (v == best[0] and (y, x) < best[1]):
                        best = (v, (y, x))
                if best is None:
                    out[(gy, gx, c)] = CellResult(FLAG_NO_COVERAGE, float("-inf"), None, None, c)
                else:
                    v, (y, x) = best
                    out[(gy, gx, c)] = CellResult(FLAG_DETERMINED, v, y, x, c)
    return out


def reference_backward(spec, cells, dy) -> np.ndarray:
    """把 pool_dy 按参照开关散回整层，得到 (H, W, C) 梯度。"""
    dy_values, _ = normalize_dy(dy, spec)
    grad = np.zeros((spec.height, spec.width, spec.channels), dtype=np.float64)
    for (gy, gx, c), r in cells.items():
        if r.flag == FLAG_DETERMINED:
            grad[r.y, r.x, c] += dy_values[gy, gx, c]
    return grad


def reference_decode(scene: Scene, rect) -> np.ndarray:
    """按整层定义把各层激活回填到原图 rect，返回 (h, w, C) FP64。"""
    y0, y1, x0, x1 = rect
    channels = max(spec.channels for spec in scene.layers.values())
    out = np.zeros((y1 - y0, x1 - x0, channels), dtype=np.float64)
    for lname in sorted(scene.layers, key=layer_sort_key):
        spec = scene.layers[lname]
        values, observed, ownable = dense_arrays(scene, lname)
        cells = reference_cells(spec, values, observed, ownable)
        oy, ox = spec.origin
        for (gy, gx, c), r in cells.items():
            if r.flag != FLAG_DETERMINED:
                continue
            Y, X = r.y + oy, r.x + ox
            if y0 <= Y < y1 and x0 <= X < x1:
                out[Y - y0, X - x0, c] += r.value
    return out


def verify_engine(engine: Engine, scene: Scene) -> None:
    """对照全图定义检验：开关、池化值、梯度落点与区域解码一致。

    要求所有瓦片均已送达（流式终态应与整层定义完全一致）。
    任何不一致都抛出 AssertionError。
    """
    for lname, spec in scene.layers.items():
        values, observed, ownable = dense_arrays(scene, lname)
        ref = reference_cells(spec, values, observed, ownable)
        for (gy, gx, c), want in ref.items():
            got = engine.evaluate_cell(lname, gy, gx, c)
            assert got.flag == want.flag, (
                f"{lname} 开关格 {(gy, gx, c)} 标记 {got.flag} != 参照 {want.flag}"
            )
            if want.flag == FLAG_DETERMINED:
                assert (got.y, got.x) == (want.y, want.x), (
                    f"{lname} 开关格 {(gy, gx, c)} 落点 {(got.y, got.x)} != 参照 {(want.y, want.x)}"
                )
                assert got.value == want.value, (
                    f"{lname} 开关格 {(gy, gx, c)} 池化值 {got.value} != 参照 {want.value}"
                )
        if lname in scene.pool_dy:
            dy = scene.pool_dy[lname]
            ref_grad = reference_backward(spec, ref, dy)
            tiles = engine.tile_gradients(lname, dy)
            for owner, tg in tiles.items():
                assert tg.deliverable, f"{lname}/{owner} 应可独立交付：{tg.waiting}"
                y0, y1, x0, x1 = tg.rect
                np.testing.assert_array_equal(
                    tg.grad, ref_grad[y0:y1, x0:x1, :],
                    err_msg=f"{lname}/{owner} 的梯度落点与整层定义不符",
                )
    for rect in scene.region_requests:
        report = engine.decode(rect)
        assert report.complete, f"区域 {rect} 应完整，未确定开关：{report.pending}"
        got = report.values
        if got.ndim == 2:
            got = got[:, :, None]
        np.testing.assert_array_equal(
            got, reference_decode(scene, rect),
            err_msg=f"区域 {rect} 的解码与整层定义不符",
        )
