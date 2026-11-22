"""整层坐标下的池化窗口几何。

窗口定义在整层坐标上：开关格 ``(gy, gx)`` 的窗口从
``(gy*stride_y - pad_top, gx*stride_x - pad_left)`` 起按空洞步进，
覆盖 kernel 个位置；越界位置按格式视为负无穷，不参与选择。

定位重叠窗口一律做算术反解，不把完整跨层展开张量放进工作区。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerSpec:
    """单层的公开几何定义（开关文件目录中同样携带）。"""

    name: str
    shape: tuple[int, int]  # 层特征图 (H, W)
    kernel: tuple[int, int]
    stride: tuple[int, int]
    dilation: tuple[int, int]
    padding: tuple[int, int]  # 上、左补边
    grid: tuple[int, int]  # 输出开关网格 (gh, gw)
    origin: tuple[int, int]  # 层坐标平移到原图的偏移
    channels: int = 1

    @property
    def height(self) -> int:
        return self.shape[0]

    @property
    def width(self) -> int:
        return self.shape[1]


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def window_positions(spec: LayerSpec, gy: int, gx: int):
    """开关格窗口覆盖的全部层坐标（含越界），按核内 (i, j) 顺序。"""
    ky, _kx = spec.kernel
    sy, sx = spec.stride
    dy, dx = spec.dilation
    pt, pl = spec.padding
    for i in range(ky):
        y = gy * sy - pt + i * dy
        for j in range(_kx):
            yield y, gx * sx - pl + j * dx


def in_bounds(spec: LayerSpec, y: int, x: int) -> bool:
    return 0 <= y < spec.height and 0 <= x < spec.width


def footprint(spec: LayerSpec, gy: int, gx: int):
    """窗口与层图像的交集 ``(y0, y1, x0, x1)``（左闭右开），无交集返回 None。

    这也是未确定开关的潜在影响范围（层坐标）；按 ``origin`` 平移即到原图。
    """
    ky, kx = spec.kernel
    sy, sx = spec.stride
    dy, dx = spec.dilation
    pt, pl = spec.padding
    y0 = max(gy * sy - pt, 0)
    x0 = max(gx * sx - pl, 0)
    y1 = min(gy * sy - pt + (ky - 1) * dy + 1, spec.height)
    x1 = min(gx * sx - pl + (kx - 1) * dx + 1, spec.width)
    if y0 >= y1 or x0 >= x1:
        return None
    return (y0, y1, x0, x1)


def cells_intersecting(spec: LayerSpec, rect):
    """窗口与层坐标矩形 ``rect=(y0, y1, x0, x1)`` 相交的开关格，行优先。"""
    y0, y1, x0, x1 = rect
    ky, kx = spec.kernel
    sy, sx = spec.stride
    dy, dx = spec.dilation
    pt, pl = spec.padding
    gh, gw = spec.grid
    gy0 = max(0, _ceil_div(y0 + pt - (ky - 1) * dy, sy))
    gy1 = min(gh - 1, (y1 - 1 + pt) // sy)
    gx0 = max(0, _ceil_div(x0 + pl - (kx - 1) * dx, sx))
    gx1 = min(gw - 1, (x1 - 1 + pl) // sx)
    for gy in range(gy0, gy1 + 1):
        for gx in range(gx0, gx1 + 1):
            yield gy, gx


def cells_covering(spec: LayerSpec, y: int, x: int):
    """窗口覆盖层坐标 ``(y, x)`` 的全部开关格。"""
    ky, kx = spec.kernel
    sy, sx = spec.stride
    dy, dx = spec.dilation
    pt, pl = spec.padding
    gh, gw = spec.grid
    gys = []
    for i in range(ky):
        num = y + pt - i * dy
        if num >= 0 and num % sy == 0:
            gy = num // sy
            if 0 <= gy < gh:
                gys.append(gy)
    gxs = []
    for j in range(kx):
        num = x + pl - j * dx
        if num >= 0 and num % sx == 0:
            gx = num // sx
            if 0 <= gx < gw:
                gxs.append(gx)
    return [(gy, gx) for gy in gys for gx in gxs]
