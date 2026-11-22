"""流式最大池化、开关出具与瓦片级反向。

开关一律使用整层坐标：瓦片的局部下标只在所有权契约划定的主记录
矩形内解释，绝不直接当作整图开关。这样接缝两侧的相同缺陷不会被
重复点亮，多次覆盖的贡献也能按整层位置求和。

单元状态三分：确定（胜出位置已最终确认）、未确定（缺片仍可能改变
胜出位置，开关不交给解码端）、无覆盖（窗口内没有合法源，值为负无穷）。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .geometry import LayerSpec, cells_intersecting, in_bounds, window_positions
from .scene import Scene, TileData

FLAG_NO_COVERAGE = 0  # 无覆盖：窗口内没有合法源，池化值为负无穷
FLAG_DETERMINED = 1  # 确定：胜出位置已最终确认
FLAG_UNDETERMINED = 2  # 未确定：缺片仍可能改变胜出位置


@dataclass(frozen=True)
class CellResult:
    """单个开关格在某一通道上的求值结果（整层坐标）。"""

    flag: int
    value: float | None  # 确定→池化值；无覆盖→负无穷；未确定→None（不出具）
    y: int | None
    x: int | None
    c: int | None


class BudgetExceeded(RuntimeError):
    """工作区预算耗尽。"""

    def __init__(self, category: str, total: int, limit: int):
        super().__init__(f"预算耗尽：{category} 记账后共用 {total} 字节，上限 {limit}")
        self.category = category
        self.total = total
        self.limit = limit


class Budget:
    """工作区预算：窗口展开、搜索路径、解码暂存分别记账。

    外部输入与请求的输出区域不计入。预算的存在是为了挡住“先展开
    完整跨层张量再定位”的实现：典型整层 3072×4096 的窗口展开远超
    budget_bytes，只有按查询惰性求值的路径能留在预算内。
    """

    WINDOW = "window"  # 窗口展开
    SEARCH = "search"  # 搜索路径
    STAGING = "staging"  # 解码暂存

    CANDIDATE_BYTES = 24  # 每个候选位：值 + 坐标 + 标记
    CELL_SEARCH_BYTES = 16  # 搜索路径上每触一格
    STAGING_SLOT_BYTES = 24  # 解码暂存每个贡献位

    def __init__(self, limit: int):
        self.limit = int(limit)
        self.used = {self.WINDOW: 0, self.SEARCH: 0, self.STAGING: 0}

    def charge(self, category: str, nbytes: int) -> None:
        if category not in self.used:
            raise KeyError(f"未知预算类别 {category!r}")
        self.used[category] += int(nbytes)
        if self.total > self.limit:
            raise BudgetExceeded(category, self.total, self.limit)

    @property
    def total(self) -> int:
        return sum(self.used.values())


class _OwnerIndex:
    """按 64×64 面元索引主记录矩形，避免逐位置扫描全部拥有者。"""

    BIN = 64

    def __init__(self, rects: dict[str, tuple[int, int, int, int]]):
        self.rects = dict(rects)
        bins: dict[tuple[int, int], list[str]] = {}
        for owner, (y0, y1, x0, x1) in self.rects.items():
            for by in range(y0 // self.BIN, (y1 - 1) // self.BIN + 1):
                for bx in range(x0 // self.BIN, (x1 - 1) // self.BIN + 1):
                    bins.setdefault((by, bx), []).append(owner)
        self.bins = bins

    def owner_of(self, y: int, x: int) -> str | None:
        for owner in self.bins.get((y // self.BIN, x // self.BIN), ()):
            y0, y1, x0, x1 = self.rects[owner]
            if y0 <= y < y1 and x0 <= x < x1:
                return owner
        return None


class _LayerState:
    def __init__(self, spec: LayerSpec, owners, bounds):
        self.spec = spec
        self.owners = dict(owners)
        self.index = _OwnerIndex(owners)
        self.bounds = dict(bounds)
        self.arrived: dict[str, TileData] = {}
        self.cache: dict[tuple[int, int, int], CellResult] = {}


@dataclass(frozen=True)
class SwitchIssue:
    """一次出具的开关快照：整层坐标，按出具时的视场（origin/grid）解释。

    快照不可变：先前交给消费端的开关索引继续按原视场解释，后续
    瓦片送达只会产生新的出具版本，不会改写旧版。
    """

    issue_id: int
    layer: str
    data_version: int
    spec: LayerSpec
    flags: np.ndarray  # (gh, gw, C) uint8
    src_y: np.ndarray  # (gh, gw, C) int32；未确定/无覆盖为 -1
    src_x: np.ndarray
    values: np.ndarray  # (gh, gw, C) float64；未确定为 NaN，无覆盖为 -inf

    def provider(self) -> "_IssueProvider":
        return _IssueProvider(self)


class _IssueProvider:
    """开关快照作为单元提供者（交接给解码端的形态）。"""

    def __init__(self, issue: SwitchIssue):
        self.issue = issue
        self.spec = issue.spec
        self.issue_id: int | None = issue.issue_id

    def cell(self, gy: int, gx: int, c: int = 0) -> CellResult:
        flag = int(self.issue.flags[gy, gx, c])
        if flag == FLAG_DETERMINED:
            return CellResult(
                flag,
                float(self.issue.values[gy, gx, c]),
                int(self.issue.src_y[gy, gx, c]),
                int(self.issue.src_x[gy, gx, c]),
                c,
            )
        if flag == FLAG_NO_COVERAGE:
            return CellResult(flag, float("-inf"), None, None, c)
        return CellResult(flag, None, None, None, c)


class _LiveProvider:
    """活引擎作为单元提供者（训练侧）。"""

    def __init__(self, engine: "Engine", layer: str):
        self.engine = engine
        self.layer = layer
        self.spec = engine.layers[layer].spec
        self.issue_id: int | None = None

    def cell(self, gy: int, gx: int, c: int = 0) -> CellResult:
        return self.engine.evaluate_cell(self.layer, gy, gx, c)


@dataclass
class TileGradient:
    """一个拥有者瓦片的梯度：可独立交付，或说明还在等什么。"""

    owner: str
    rect: tuple[int, int, int, int]
    grad: np.ndarray  # (h, w, C) float64；未送齐时仅为部分和
    deliverable: bool
    waiting: list  # ("switch", gy, gx, c) 等开关确认；("dy", gy, gx, c) 等反向贡献


def normalize_dy(dy, spec: LayerSpec) -> tuple[np.ndarray, np.ndarray]:
    """把 pool_dy 归一为 (值 (gh,gw,C), 已送达 (gh,gw))。

    接受整网格数组或 ``{(gy, gx): 值}`` 的部分送达形式。
    """
    gh, gw = spec.grid
    channels = spec.channels
    if isinstance(dy, dict):
        values = np.zeros((gh, gw, channels))
        arrived = np.zeros((gh, gw), dtype=bool)
        for key, v in dy.items():
            gy, gx = int(key[0]), int(key[1])
            vec = np.asarray(v, dtype=np.float64).reshape(-1)
            if vec.size != channels:
                raise ValueError(f"pool_dy[{key}] 通道数应为 {channels}")
            values[gy, gx] = vec
            arrived[gy, gx] = True
        return values, arrived
    arr = np.asarray(dy, dtype=np.float64)
    if arr.ndim == 2:
        if channels != 1:
            raise ValueError("多通道层的 pool_dy 需为 (gh, gw, C)")
        arr = arr[:, :, None]
    if arr.shape != (gh, gw, channels):
        raise ValueError(f"pool_dy 形状 {arr.shape} 应为 {(gh, gw, channels)}")
    return arr, np.ones((gh, gw), dtype=bool)


class Engine:
    """流式池化引擎：按时间线吸收瓦片，惰性求值开关格。"""

    def __init__(self, scene: Scene, budget: Budget | None = None):
        self.scene = scene
        self.budget = budget
        self.layers = {
            lname: _LayerState(spec, scene.primary[lname], scene.bounds.get(lname, {}))
            for lname, spec in scene.layers.items()
        }
        self.pool_dy = dict(scene.pool_dy)
        self._issue_counter = itertools.count(1)

    @classmethod
    def from_scene(cls, scene: Scene, budget_bytes: int | None = None) -> "Engine":
        limit = scene.budget_bytes if budget_bytes is None else budget_bytes
        return cls(scene, Budget(limit) if limit is not None else None)

    # ---- 瓦片送达 ----------------------------------------------------

    def deliver_tile(self, layer: str, owner: str, values=None, observed=None, offset=None):
        """送达一个拥有者的记录。

        只取所有权契约划定的主记录范围；记录可以带光环（重叠读取），
        主记录之外的部分仅是邻接线索，不参与最终选择。
        """
        st = self.layers[layer]
        rect = st.owners[owner]
        y0, y1, x0, x1 = rect
        h, w = y1 - y0, x1 - x0
        if values is None:
            tile = self.scene.tiles[layer][owner]
            values, observed = tile.values, tile.observed
            offset = (y0, x0)
        else:
            values = np.asarray(values, dtype=np.float64)
            if values.ndim == 2:
                values = values[:, :, None]
            if observed is None:
                observed = np.ones(values.shape[:2], dtype=bool)
            if offset is None:
                offset = (y0, x0)
        observed = np.asarray(observed, dtype=bool)
        if observed.shape != values.shape[:2]:
            raise ValueError("observed 与 values 的形状不一致")
        if values.shape[2] != st.spec.channels:
            raise ValueError(f"通道数应为 {st.spec.channels}")
        oy, ox = offset
        vh, vw = values.shape[0], values.shape[1]
        if not (oy <= y0 and ox <= x0 and oy + vh >= y1 and ox + vw >= x1):
            raise ValueError(f"提交的记录未覆盖主记录矩形 {rect}")
        clipped_v = np.full((h, w, st.spec.channels), np.nan)
        clipped_o = np.zeros((h, w), dtype=bool)
        sy0, sx0 = y0 - oy, x0 - ox
        clipped_v[:] = values[sy0 : sy0 + h, sx0 : sx0 + w]
        clipped_o[:] = observed[sy0 : sy0 + h, sx0 : sx0 + w]
        if not np.isfinite(clipped_v[clipped_o]).all():
            raise ValueError("已观测位置的取值必须有限")
        st.arrived[owner] = TileData(clipped_v, clipped_o)
        st.cache.clear()

    # ---- 单元求值 ----------------------------------------------------

    def evaluate_cell(self, layer: str, gy: int, gx: int, channel: int = 0) -> CellResult:
        st = self.layers[layer]
        key = (gy, gx, channel)
        hit = st.cache.get(key)
        if hit is not None:
            return hit
        if self.budget is not None:
            ky, kx = st.spec.kernel
            self.budget.charge(Budget.WINDOW, ky * kx * Budget.CANDIDATE_BYTES)
        result = self._evaluate(st, gy, gx, channel)
        st.cache[key] = result
        return result

    def _evaluate(self, st: _LayerState, gy: int, gx: int, c: int) -> CellResult:
        spec = st.spec
        best_v = best_y = best_x = None
        missing: dict[str, list[tuple[int, int]]] = {}
        for y, x in window_positions(spec, gy, gx):
            if not in_bounds(spec, y, x):
                continue  # 图外补齐按负无穷处理
            owner = st.index.owner_of(y, x)
            if owner is None:
                continue  # 非任何主记录：不是合法源
            tile = st.arrived.get(owner)
            if tile is None:
                missing.setdefault(owner, []).append((y, x))  # 缺测：整片未送达
                continue
            rect = st.owners[owner]
            ly, lx = y - rect[0], x - rect[2]
            if not tile.observed[ly, lx]:
                continue  # 已确认不可观测，排除
            v = tile.values[ly, lx, c]
            if best_v is None or v > best_v or (v == best_v and (y, x) < (best_y, best_x)):
                best_v, best_y, best_x = v, y, x
        if not missing:
            if best_v is None:
                return CellResult(FLAG_NO_COVERAGE, float("-inf"), None, None, c)
            return CellResult(FLAG_DETERMINED, best_v, best_y, best_x, c)
        bounds = [st.bounds.get(o) for o in missing]
        if best_v is None or any(b is None for b in bounds):
            # 上界未知（或未声明）：缺片仍可能改变胜出位置
            return CellResult(FLAG_UNDETERMINED, None, None, None, c)
        bmax = max(bounds)
        if bmax < best_v:
            return CellResult(FLAG_DETERMINED, best_v, best_y, best_x, c)
        if bmax > best_v:
            return CellResult(FLAG_UNDETERMINED, None, None, None, c)
        # 缺片上界与当前最高持平：仍按 (y, x, 通道) 决胜
        for owner, positions in missing.items():
            if st.bounds.get(owner) != bmax:
                continue
            for y, x in positions:
                if (y, x) < (best_y, best_x):
                    return CellResult(FLAG_UNDETERMINED, None, None, None, c)
        return CellResult(FLAG_DETERMINED, best_v, best_y, best_x, c)

    # ---- 开关出具 ----------------------------------------------------

    def issue_switches(self, layer: str) -> SwitchIssue:
        """按当前状态出具整层开关快照；未确定的格不出具开关。"""
        st = self.layers[layer]
        spec = st.spec
        gh, gw = spec.grid
        channels = spec.channels
        flags = np.zeros((gh, gw, channels), dtype=np.uint8)
        src_y = np.full((gh, gw, channels), -1, dtype=np.int32)
        src_x = np.full((gh, gw, channels), -1, dtype=np.int32)
        values = np.full((gh, gw, channels), np.nan)
        for gy in range(gh):
            for gx in range(gw):
                for c in range(channels):
                    r = self.evaluate_cell(layer, gy, gx, c)
                    flags[gy, gx, c] = r.flag
                    if r.flag == FLAG_DETERMINED:
                        src_y[gy, gx, c] = r.y
                        src_x[gy, gx, c] = r.x
                        values[gy, gx, c] = r.value
                    elif r.flag == FLAG_NO_COVERAGE:
                        values[gy, gx, c] = -np.inf
        return SwitchIssue(
            next(self._issue_counter), layer, self.scene.data_version, spec,
            flags, src_y, src_x, values,
        )

    def provider(self, layer: str) -> _LiveProvider:
        return _LiveProvider(self, layer)

    def providers(self) -> dict[str, _LiveProvider]:
        return {lname: self.provider(lname) for lname in self.layers}

    # ---- 区域解码 / 反向（共用路径的入口） ---------------------------

    def decode(self, rect, providers=None, pooled_values=None):
        from .region import decode_region

        if providers is None:
            providers = self.providers()
        return decode_region(rect, providers, pooled_values, self.budget)

    def backward(self, rect, providers=None, pool_dy=None, registry=None):
        from .region import backward_region

        if providers is None:
            providers = self.providers()
        dy = self.pool_dy if pool_dy is None else pool_dy
        return backward_region(rect, providers, dy, registry, self.budget)

    # ---- 瓦片级反向 ----------------------------------------------------

    def provide_pool_dy(self, layer: str, dy) -> None:
        self.pool_dy[layer] = normalize_dy(dy, self.layers[layer].spec)[0]

    def tile_gradients(self, layer: str, dy=None, registry=None) -> dict[str, TileGradient]:
        """把各有效窗口的贡献归回真正拥有者。

        瓦片可独立交付的条件：窗口与其主记录矩形相交的开关格全部
        已确定，且这些格的反向贡献已送齐；否则在 waiting 中说明。
        """
        st = self.layers[layer]
        spec = st.spec
        if dy is None:
            if layer not in self.pool_dy:
                raise ValueError(f"层 {layer} 尚未提供 pool_dy")
            dy = self.pool_dy[layer]
        dy_values, dy_arrived = normalize_dy(dy, spec)
        result = {}
        for owner, rect in st.owners.items():
            grad = np.zeros((rect[1] - rect[0], rect[3] - rect[2], spec.channels))
            waiting: list = []
            for gy, gx in cells_intersecting(spec, rect):
                if self.budget is not None:
                    self.budget.charge(Budget.SEARCH, Budget.CELL_SEARCH_BYTES)
                for c in range(spec.channels):
                    r = self.evaluate_cell(layer, gy, gx, c)
                    if r.flag == FLAG_UNDETERMINED:
                        waiting.append(("switch", gy, gx, c))
                        continue
                    if r.flag != FLAG_DETERMINED:
                        continue
                    if not (rect[0] <= r.y < rect[1] and rect[2] <= r.x < rect[3]):
                        continue  # 贡献属于别的拥有者
                    if not dy_arrived[gy, gx]:
                        waiting.append(("dy", gy, gx, c))
                        continue
                    if self.budget is not None:
                        self.budget.charge(Budget.STAGING, Budget.STAGING_SLOT_BYTES)
                    grad[r.y - rect[0], r.x - rect[2], c] += dy_values[gy, gx, c]
                    if registry is not None:
                        registry.record(layer, None, gy, gx, c, float(dy_values[gy, gx, c]))
            result[owner] = TileGradient(owner, rect, grad, not waiting, waiting)
        return result

    # ---- 时间线 ----------------------------------------------------

    def handle(self, event):
        kind = event[0]
        if kind == "tile":
            _, layer, owner = event
            self.deliver_tile(layer, owner)
            return None
        if kind == "region":
            _, idx = event
            return self.decode(self.scene.region_requests[idx])
        raise ValueError(f"未知事件 {event!r}")
