"""检查交换资料的形状和索引。"""

import argparse
import json
from pathlib import Path

import numpy as np


def inspect(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    validate(data)
    return {
        "资料": path.name,
        "字段": sorted(data),
        "字节": path.stat().st_size,
        "数值环境": np.__version__,
    }


def validate(data: dict) -> None:
    for layer, shape in data["layers"].items():
            for owner, bounds in data["primary"][layer].items():
                y0, y1, x0, x1 = bounds
                tile = data["features"][layer][owner]
                assert 0 <= y0 < y1 <= shape["shape"][0] and 0 <= x0 < x1 <= shape["shape"][1]
                assert len(tile) == y1-y0 and all(len(line) == x1-x0 for line in tile)
            assert list(np.shape(data["pool_dy"][layer])) == shape["grid"]


def main() -> None:
    parser = argparse.ArgumentParser(description="检查本地交换资料")
    parser.add_argument("scene", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect(args.scene), ensure_ascii=False))


if __name__ == "__main__":
    main()
