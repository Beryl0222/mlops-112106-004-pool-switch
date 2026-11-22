"""交换资料（JSON）的解析、校验与内存表示。

原始图（层形状与原图偏移）、有效视场（`primary` 所有权契约）和
特征值分开提供；`null` 是已确认不可观测，整个来源未送达是缺测，
两者分开。未声明边界的格按图面条件处理，不以空数组补零。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import LayerSpec


@dataclass
class TileData:
    """一个拥有者提交的特征记录：数值与“已观测”掩码分开。"""

    values: np.ndarray  # (h, w, C) float64；未观测位置为 NaN
    observed: np.ndarray  # (h, w) bool；False 表示已确认不可观测（文件中的 null）


@dataclass
class Scene:
    layers: dict[str, LayerSpec]
    primary: dict[str, dict[str, tuple[int, int, int, int]]]
    tiles: dict[str, dict[str, TileData]]
    bounds: dict[str, dict[str, float | None]]
    pool_dy: dict[str, np.ndarray]  # (gh, gw, C) float64
    region_requests: list[tuple[int, int, int, int]]
    timeline: list[tuple]
    budget_bytes: int | None
    data_version: int = 0


def _as_pair(value, what: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{what} 应为二元组，得到 {value!r}")
    return int(value[0]), int(value[1])


def _as_rect(value, what: str) -> tuple[int, int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{what} 应为四元矩形，得到 {value!r}")
    y0, y1, x0, x1 = (int(v) for v in value)
    if not (0 <= y0 < y1 and 0 <= x0 < x1):
        raise ValueError(f"{what} 矩形非法：{value!r}")
    return y0, y1, x0, x1


def _check_disjoint(rects: dict[str, tuple[int, int, int, int]], what: str) -> None:
    items = sorted(rects.items())
    for i, (a_name, a) in enumerate(items):
        for b_name, b in items[i + 1 :]:
            if a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]:
                raise ValueError(f"{what} 中 {a_name} 与 {b_name} 的主记录矩形重叠")


def _parse_tile(raw, rect, what: str) -> tuple[TileData, int]:
    """把文件中的特征记录解析为 (TileData, 通道数)。

    支持 (h, w) 单通道与 (h, w, C) 通道在末维两种形态；null 只能
    整位置出现（该位置所有通道同时为 null），表示已确认不可观测。
    """
    h, w = rect[1] - rect[0], rect[3] - rect[2]
    if not isinstance(raw, (list, tuple)) or len(raw) != h:
        raise ValueError(f"{what} 行数应为 {h}")
    rows = [list(row) for row in raw]
    for row in rows:
        if len(row) != w:
            raise ValueError(f"{what} 列数应为 {w}")
    channels = 1
    if h and w and isinstance(rows[0][0], (list, tuple)):
        channels = len(rows[0][0])
    values = np.full((h, w, channels), np.nan)
    observed = np.ones((h, w), dtype=bool)
    for i in range(h):
        for j in range(w):
            cell = rows[i][j]
            vec = list(cell) if isinstance(cell, (list, tuple)) else [cell]
            if len(vec) != channels:
                raise ValueError(f"{what} 位置 {(i, j)} 通道数不一致")
            if all(v is None for v in vec):
                observed[i, j] = False
                continue
            if any(v is None for v in vec):
                raise ValueError(f"{what} 位置 {(i, j)} 的 null 未整位置出现")
            for k, v in enumerate(vec):
                f = float(v)
                if not np.isfinite(f):
                    raise ValueError(f"{what} 位置 {(i, j)} 取值必须有限")
                values[i, j, k] = f
    return TileData(values, observed), channels


def parse_scene(data: dict) -> Scene:
    for key in ("layers", "primary", "features", "region_requests", "timeline"):
        if key not in data:
            raise KeyError(key)
    raw_layers = data["layers"]
    features = data["features"]

    primary: dict[str, dict[str, tuple[int, int, int, int]]] = {}
    tiles: dict[str, dict[str, TileData]] = {}
    channels: dict[str, int] = {}
    for lname, owners in data["primary"].items():
        if lname not in raw_layers:
            raise KeyError(f"primary 引用了未知层 {lname!r}")
        shape = _as_pair(raw_layers[lname]["shape"], f"layers[{lname}].shape")
        rects = {}
        for owner, raw_rect in owners.items():
            rect = _as_rect(raw_rect, f"primary[{lname}][{owner}]")
            if rect[1] > shape[0] or rect[3] > shape[1]:
                raise ValueError(f"primary[{lname}][{owner}] 越出层形状 {shape}")
            rects[owner] = rect
        _check_disjoint(rects, f"primary[{lname}]")
        primary[lname] = rects
        layer_tiles = {}
        for owner, rect in rects.items():
            try:
                raw = features[lname][owner]
            except KeyError:
                raise KeyError(f"缺少特征记录 features[{lname}][{owner}]") from None
            tile, c = _parse_tile(raw, rect, f"features[{lname}][{owner}]")
            if lname in channels and channels[lname] != c:
                raise ValueError(f"层 {lname} 的通道数不一致")
            channels[lname] = c
            layer_tiles[owner] = tile
        tiles[lname] = layer_tiles

    layers: dict[str, LayerSpec] = {}
    for lname, raw in raw_layers.items():
        spec = LayerSpec(
            name=lname,
            shape=_as_pair(raw["shape"], f"layers[{lname}].shape"),
            kernel=_as_pair(raw["kernel"], f"layers[{lname}].kernel"),
            stride=_as_pair(raw["stride"], f"layers[{lname}].stride"),
            dilation=_as_pair(raw["dilation"], f"layers[{lname}].dilation"),
            padding=_as_pair(raw["padding"], f"layers[{lname}].padding"),
            grid=_as_pair(raw["grid"], f"layers[{lname}].grid"),
            origin=_as_pair(raw["origin"], f"layers[{lname}].origin"),
            channels=channels.get(lname, 1),
        )
        for field_name in ("kernel", "stride", "dilation", "grid", "shape"):
            a, b = getattr(spec, field_name)
            if a < 1 or b < 1:
                raise ValueError(f"layers[{lname}].{field_name} 必须为正")
        if spec.padding[0] < 0 or spec.padding[1] < 0:
            raise ValueError(f"layers[{lname}].padding 不能为负")
        layers[lname] = spec

    bounds: dict[str, dict[str, float | None]] = {}
    for lname, per_owner in data.get("bounds", {}).items():
        if lname not in primary:
            raise KeyError(f"bounds 引用了未知层 {lname!r}")
        layer_bounds = {}
        for owner, bound in per_owner.items():
            if owner not in primary[lname]:
                raise KeyError(f"bounds 引用了未知拥有者 {lname}/{owner}")
            layer_bounds[owner] = None if bound is None else float(bound)
        bounds[lname] = layer_bounds

    pool_dy: dict[str, np.ndarray] = {}
    for lname, raw in data.get("pool_dy", {}).items():
        if lname not in layers:
            raise KeyError(f"pool_dy 引用了未知层 {lname!r}")
        arr = np.asarray(raw, dtype=np.float64)
        gh, gw = layers[lname].grid
        c = layers[lname].channels
        if arr.ndim == 2 and c == 1:
            arr = arr[:, :, None]
        if arr.shape != (gh, gw, c):
            raise ValueError(f"pool_dy[{lname}] 形状 {arr.shape} 应为 {(gh, gw, c)}")
        pool_dy[lname] = arr

    region_requests = [
        _as_rect(r, f"region_requests[{i}]") for i, r in enumerate(data["region_requests"])
    ]

    timeline = []
    for ev in data["timeline"]:
        ev = tuple(ev)
        if not ev:
            raise ValueError("timeline 含空事件")
        if ev[0] == "tile":
            if len(ev) != 3 or ev[1] not in layers or ev[2] not in primary.get(ev[1], {}):
                raise ValueError(f"非法瓦片事件 {ev!r}")
            timeline.append(("tile", ev[1], ev[2]))
        elif ev[0] == "region":
            if len(ev) != 2 or not 0 <= int(ev[1]) < len(region_requests):
                raise ValueError(f"非法区域事件 {ev!r}")
            timeline.append(("region", int(ev[1])))
        else:
            raise ValueError(f"未知事件 {ev!r}")

    budget = data.get("budget_bytes")
    return Scene(
        layers=layers,
        primary=primary,
        tiles=tiles,
        bounds=bounds,
        pool_dy=pool_dy,
        region_requests=region_requests,
        timeline=timeline,
        budget_bytes=None if budget is None else int(budget),
        data_version=int(data.get("version", 0)),
    )


def load_scene(path) -> Scene:
    return parse_scene(json.loads(Path(path).read_text(encoding="utf-8")))
