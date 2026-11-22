"""瓦片池化与开关交接：边端与训练共用的 Python 张量路径。"""

from .engine import (
    FLAG_DETERMINED,
    FLAG_NO_COVERAGE,
    FLAG_UNDETERMINED,
    Budget,
    BudgetExceeded,
    CellResult,
    Engine,
    SwitchIssue,
    TileGradient,
    normalize_dy,
)
from .geometry import (
    LayerSpec,
    cells_covering,
    cells_intersecting,
    footprint,
    in_bounds,
    window_positions,
)
from .reference import (
    dense_arrays,
    reference_backward,
    reference_cells,
    reference_decode,
    verify_engine,
)
from .region import (
    GradientRegistry,
    RegionReport,
    backward_region,
    decode_region,
    layer_sort_key,
)
from .scene import Scene, TileData, load_scene, parse_scene
from .switchfile import SwitchFile, write_switch_file

__all__ = [
    "FLAG_DETERMINED",
    "FLAG_NO_COVERAGE",
    "FLAG_UNDETERMINED",
    "Budget",
    "BudgetExceeded",
    "CellResult",
    "Engine",
    "GradientRegistry",
    "LayerSpec",
    "RegionReport",
    "Scene",
    "SwitchFile",
    "SwitchIssue",
    "TileData",
    "TileGradient",
    "backward_region",
    "cells_covering",
    "cells_intersecting",
    "decode_region",
    "dense_arrays",
    "footprint",
    "in_bounds",
    "layer_sort_key",
    "load_scene",
    "normalize_dy",
    "parse_scene",
    "reference_backward",
    "reference_cells",
    "reference_decode",
    "verify_engine",
    "window_positions",
    "write_switch_file",
]
