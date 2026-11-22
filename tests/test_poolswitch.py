"""端到端校验：流式池化、开关交接、区域解码与反向回填。"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from poolswitch import (
    FLAG_DETERMINED,
    FLAG_NO_COVERAGE,
    FLAG_UNDETERMINED,
    Budget,
    BudgetExceeded,
    Engine,
    GradientRegistry,
    LayerSpec,
    SwitchFile,
    SwitchIssue,
    cells_covering,
    cells_intersecting,
    decode_region,
    dense_arrays,
    load_scene,
    parse_scene,
    verify_engine,
    window_positions,
    write_switch_file,
)

ROOT = Path(__file__).resolve().parents[1]


def load_sample(name):
    return load_scene(ROOT / "samples" / name)


def run_tiles(engine, scene):
    for event in scene.timeline:
        engine.handle(event)


class GeometryTests(unittest.TestCase):
    """窗口定位只做算术反解，与暴力展开一致。"""

    def test_cells_covering_matches_brute_force(self):
        spec = LayerSpec("t", (5, 7), (2, 3), (2, 2), (1, 2), (1, 1), (3, 4), (0, 0))
        for y in range(5):
            for x in range(7):
                want = set()
                for gy in range(3):
                    for gx in range(4):
                        if (y, x) in set(window_positions(spec, gy, gx)):
                            want.add((gy, gx))
                self.assertEqual(set(cells_covering(spec, y, x)), want)

    def test_cells_intersecting_matches_brute_force(self):
        spec = LayerSpec("t", (5, 7), (2, 3), (2, 2), (1, 2), (1, 1), (3, 4), (0, 0))
        rect = (1, 4, 2, 6)
        want = set()
        for gy in range(3):
            for gx in range(4):
                if any(rect[0] <= y < rect[1] and rect[2] <= x < rect[3]
                       for y, x in window_positions(spec, gy, gx)):
                    want.add((gy, gx))
        self.assertEqual(set(cells_intersecting(spec, rect)), want)


class ForwardTimelineTests(unittest.TestCase):
    """按时间线吸收瓦片，区域报告与手工核算一致。"""

    def test_scene_timeline_reports(self):
        scene = load_sample("scene.json")
        engine = Engine.from_scene(scene)
        reports = [engine.handle(event) for event in scene.timeline]
        r3, r5, r7 = reports[2], reports[4], reports[6]

        # 事件 3：l0/l1 右片未到，区域 [0,2)×[1,3) 不完整
        self.assertFalse(r3.complete)
        np.testing.assert_array_equal(r3.values, [[0.0, 0.0], [4.0, 4.0]])
        self.assertEqual(len(r3.pending), 5)

        # 事件 5：缺片只报在 l0，但未定开关的窗口命中本区域 ⇒ 不能冒充完整
        self.assertFalse(r5.complete)
        np.testing.assert_array_equal(r5.values, np.zeros((2, 2)))
        self.assertEqual(len(r5.pending), 4)
        self.assertTrue(all(layer == "l0" for layer, *_ in r5.pending))

        # 事件 7：全部送达后同一区域完整
        self.assertTrue(r7.complete)
        np.testing.assert_array_equal(r7.values, [[0.0, 0.0], [8.0, 8.0]])

    def test_edge_ties_bound_equal_tiebreak(self):
        scene = load_sample("edge-ties.json")
        engine = Engine.from_scene(scene)
        engine.handle(scene.timeline[0])
        engine.handle(scene.timeline[1])

        # 上界与当前最高持平，挑战位 (0,2) 排在 (1,0) 前 ⇒ 未确定
        r = engine.evaluate_cell("l1", 0, 1)
        self.assertEqual(r.flag, FLAG_UNDETERMINED)
        self.assertIsNone(r.value)  # 未确定的候选再大也不出具

        # 挑战位都排在 (1,0) 后 ⇒ 按 (y, x, 通道) 决胜，确定
        r = engine.evaluate_cell("l1", 1, 1)
        self.assertEqual((r.flag, r.y, r.x, r.value), (FLAG_DETERMINED, 1, 0, 4.0))

        report = engine.handle(scene.timeline[2])  # 区域 0：不完整
        self.assertFalse(report.complete)
        np.testing.assert_array_equal(report.values, [[0.0, 0.0]])

        engine.handle(scene.timeline[3])  # l1 右片
        report = engine.handle(scene.timeline[4])  # 区域 1：l0 仍缺
        self.assertFalse(report.complete)
        np.testing.assert_array_equal(report.values, [[8.0, 8.0]])

        engine.handle(scene.timeline[5])  # l0 右片
        report = engine.handle(scene.timeline[6])  # 区域 0：完整
        self.assertTrue(report.complete)
        np.testing.assert_array_equal(report.values, [[0.0, 0.0]])

    def test_tie_breaks_by_whole_layer_coordinates(self):
        scene = load_sample("scene.json")
        engine = Engine.from_scene(scene)
        run_tiles(engine, scene)
        # l0 (1,1)：3 平在 (2,2) 与 (2,3)，取全层坐标较小者
        r = engine.evaluate_cell("l0", 1, 1)
        self.assertEqual((r.flag, r.y, r.x, r.value), (FLAG_DETERMINED, 2, 2, 3.0))
        # l1 (1,2)：2 平在 (1,1) 与 (2,3)，取 (1,1)
        r = engine.evaluate_cell("l1", 1, 2)
        self.assertEqual((r.y, r.x), (1, 1))


class ValueSemanticsTests(unittest.TestCase):
    """图外补齐、缺测、不可观测与真值零分开解释。"""

    def make_engine(self, with_bounds=True):
        data = {
            "layers": {"m": {"shape": [2, 2], "kernel": [1, 2], "stride": [1, 1],
                             "dilation": [1, 1], "padding": [0, 0], "grid": [2, 2],
                             "origin": [0, 0]}},
            "primary": {"m": {"solo": [0, 2, 0, 2]}},
            "features": {"m": {"solo": [[0, None], [-3, -5]]}},
            "pool_dy": {"m": [[0, 0], [0, 0]]},
            "region_requests": [],
            "timeline": [],
        }
        if with_bounds:
            data["bounds"] = {"m": {"solo": 10}}
        scene = parse_scene(data)
        engine = Engine.from_scene(scene)
        return engine

    def test_zero_negative_null_and_oob(self):
        engine = self.make_engine()
        engine.deliver_tile("m", "solo")
        # 窗口 {0, null}：真值 0 合法且胜出，落点 (0,0)
        r = engine.evaluate_cell("m", 0, 0)
        self.assertEqual((r.flag, r.y, r.x, r.value), (FLAG_DETERMINED, 0, 0, 0.0))
        # 窗口 {null, 越界}：没有合法源 ⇒ 无覆盖，值为负无穷
        r = engine.evaluate_cell("m", 0, 1)
        self.assertEqual((r.flag, r.value), (FLAG_NO_COVERAGE, float("-inf")))
        # 池化值可以为负
        r = engine.evaluate_cell("m", 1, 0)
        self.assertEqual((r.flag, r.value), (FLAG_DETERMINED, -3.0))
        r = engine.evaluate_cell("m", 1, 1)
        self.assertEqual((r.flag, r.value), (FLAG_DETERMINED, -5.0))

    def test_missing_tile_is_not_zero_filled(self):
        data = {
            "layers": {"m": {"shape": [1, 4], "kernel": [1, 2], "stride": [1, 2],
                             "dilation": [1, 1], "padding": [0, 0], "grid": [1, 2],
                             "origin": [0, 0]}},
            "primary": {"m": {"left": [0, 1, 0, 2], "right": [0, 1, 2, 4]}},
            "features": {"m": {"left": [[1.0, 2.0]], "right": [[3.0, 4.0]]}},
            "pool_dy": {"m": [[0.0, 0.0]]},
            "region_requests": [],
            "timeline": [],
        }
        engine = Engine.from_scene(parse_scene(data))
        engine.deliver_tile("m", "left")
        # 右片缺测且未声明边界：上界未知 ⇒ 未确定，绝不按零补
        r = engine.evaluate_cell("m", 0, 1)
        self.assertEqual(r.flag, FLAG_UNDETERMINED)
        engine.deliver_tile("m", "right")
        r = engine.evaluate_cell("m", 0, 1)
        self.assertEqual((r.flag, r.y, r.x, r.value), (FLAG_DETERMINED, 0, 3, 4.0))

    def test_bound_allows_partial_determination(self):
        def make(bound):
            data = {
                "layers": {"m": {"shape": [1, 4], "kernel": [1, 3], "stride": [1, 2],
                                 "dilation": [1, 1], "padding": [0, 1], "grid": [1, 2],
                                 "origin": [0, 0]}},
                "primary": {"m": {"left": [0, 1, 0, 2], "right": [0, 1, 2, 4]}},
                "features": {"m": {"left": [[1.0, 2.0]], "right": [[3.0, 4.0]]}},
                "bounds": {"m": {"right": bound}},
                "pool_dy": {"m": [[0.0, 0.0]]},
                "region_requests": [],
                "timeline": [],
            }
            engine = Engine.from_scene(parse_scene(data))
            engine.deliver_tile("m", "left")
            return engine

        # 窗口 x{1,2,3}：已到候选 2@(0,1)，缺片在右片
        # 上界压不过当前最高 ⇒ 确定
        r = make(1.0).evaluate_cell("m", 0, 1)
        self.assertEqual((r.flag, r.y, r.x, r.value), (FLAG_DETERMINED, 0, 1, 2.0))
        # 上界持平，挑战位 (0,2)、(0,3) 都排在 (0,1) 后 ⇒ 仍确定
        r = make(2.0).evaluate_cell("m", 0, 1)
        self.assertEqual((r.flag, r.y, r.x), (FLAG_DETERMINED, 0, 1))
        # 上界更高 ⇒ 未确定，候选再大也不出具
        r = make(100.0).evaluate_cell("m", 0, 1)
        self.assertEqual((r.flag, r.value), (FLAG_UNDETERMINED, None))
        # 上界为空 ⇒ 未知 ⇒ 未确定
        r = make(None).evaluate_cell("m", 0, 1)
        self.assertEqual(r.flag, FLAG_UNDETERMINED)

    def test_overlapping_records_are_only_clues(self):
        data = {
            "layers": {"m": {"shape": [1, 4], "kernel": [1, 2], "stride": [1, 2],
                             "dilation": [1, 1], "padding": [0, 0], "grid": [1, 2],
                             "origin": [0, 0]}},
            "primary": {"m": {"left": [0, 1, 0, 2], "right": [0, 1, 2, 4]}},
            "features": {"m": {"left": [[1.0, 2.0]], "right": [[3.0, 4.0]]}},
            "bounds": {"m": {"right": 100.0}},
            "pool_dy": {"m": [[0.0, 0.0]]},
            "region_requests": [],
            "timeline": [],
        }
        engine = Engine.from_scene(parse_scene(data))
        # left 提交带光环的重叠记录：主记录之外的部分只是邻接线索
        engine.deliver_tile("m", "left", values=np.array([[1.0, 2.0, 999.0, 888.0]]),
                            offset=(0, 0))
        r = engine.evaluate_cell("m", 0, 1)
        self.assertEqual(r.flag, FLAG_UNDETERMINED)  # 光环里的 999 不参与选择
        engine.deliver_tile("m", "right")
        r = engine.evaluate_cell("m", 0, 1)
        self.assertEqual((r.flag, r.y, r.x, r.value), (FLAG_DETERMINED, 0, 3, 4.0))
        r = engine.evaluate_cell("m", 0, 0)
        self.assertEqual((r.y, r.x, r.value), (0, 1, 2.0))

    def test_channels_tie_break_and_per_channel_switches(self):
        data = {
            "layers": {"m": {"shape": [1, 2], "kernel": [1, 2], "stride": [1, 2],
                             "dilation": [1, 1], "padding": [0, 0], "grid": [1, 1],
                             "origin": [0, 0]}},
            "primary": {"m": {"solo": [0, 1, 0, 2]}},
            "features": {"m": {"solo": [[[5.0, 1.0], [5.0, 2.0]]]}},
            "pool_dy": {"m": [[[1.0, 10.0]]]},
            "region_requests": [],
            "timeline": [],
        }
        scene = parse_scene(data)
        self.assertEqual(scene.layers["m"].channels, 2)
        engine = Engine.from_scene(scene)
        engine.deliver_tile("m", "solo")
        r0 = engine.evaluate_cell("m", 0, 0, 0)
        r1 = engine.evaluate_cell("m", 0, 0, 1)
        self.assertEqual((r0.y, r0.x, r0.value), (0, 0, 5.0))  # 通道内平局取坐标较小者
        self.assertEqual((r1.y, r1.x, r1.value), (0, 1, 2.0))
        tiles = engine.tile_gradients("m")
        self.assertEqual(tiles["solo"].grad[0, 0, 0], 1.0)
        self.assertEqual(tiles["solo"].grad[0, 1, 1], 10.0)


class ReferenceTests(unittest.TestCase):
    """流式终态对照全图定义。"""

    def test_verify_both_samples(self):
        for name in ("scene.json", "edge-ties.json"):
            scene = load_sample(name)
            engine = Engine.from_scene(scene)
            run_tiles(engine, scene)
            verify_engine(engine, scene)  # 不抛即通过

    def test_retiling_keeps_whole_image_semantics(self):
        scene = load_sample("scene.json")
        base = Engine.from_scene(scene)
        run_tiles(base, scene)

        # 换瓦片划分：l0 切三片，l1 按行切两片，特征从整层稠密张量重新切片
        def slice_features(values, observed, rect):
            y0, y1, x0, x1 = rect
            return [
                [None if not observed[i, j] else float(values[i, j, 0])
                 for j in range(x0, x1)]
                for i in range(y0, y1)
            ]

        data = json.loads((ROOT / "samples" / "scene.json").read_text(encoding="utf-8"))
        full = {lname: dense_arrays(scene, lname) for lname in scene.layers}
        l0_v, l0_o, _ = full["l0"]
        l1_v, l1_o, _ = full["l1"]
        data["primary"] = {
            "l0": {"a": [0, 3, 0, 1], "b": [0, 3, 1, 3], "c": [0, 3, 3, 5]},
            "l1": {"u": [0, 2, 0, 4], "v": [2, 3, 0, 4]},
        }
        data["features"] = {
            "l0": {
                "a": slice_features(l0_v, l0_o, (0, 3, 0, 1)),
                "b": slice_features(l0_v, l0_o, (0, 3, 1, 3)),
                "c": slice_features(l0_v, l0_o, (0, 3, 3, 5)),
            },
            "l1": {
                "u": slice_features(l1_v, l1_o, (0, 2, 0, 4)),
                "v": slice_features(l1_v, l1_o, (2, 3, 0, 4)),
            },
        }
        data["bounds"] = {}
        data["timeline"] = [
            ["tile", "l0", "c"], ["tile", "l1", "v"], ["tile", "l0", "a"],
            ["tile", "l1", "u"], ["tile", "l0", "b"],
            ["region", 0], ["region", 1],
        ]
        scene2 = parse_scene(data)
        engine2 = Engine.from_scene(scene2)
        reports2 = [engine2.handle(event) for event in scene2.timeline]
        verify_engine(engine2, scene2)

        # 换划分、换请求顺序后，同一激活与梯度落点一致
        for rect in scene.region_requests:
            want = base.decode(rect)
            got = engine2.decode(rect)
            self.assertTrue(got.complete)
            np.testing.assert_array_equal(got.values, want.values)
        whole = (0, 3, 0, 6)
        np.testing.assert_array_equal(base.backward(whole).values,
                                      engine2.backward(whole).values)
        self.assertTrue(reports2[-1].complete)


class SwitchIssueTests(unittest.TestCase):
    def test_issued_switches_are_immutable_and_keep_fov(self):
        scene = load_sample("scene.json")
        engine = Engine.from_scene(scene)
        engine.handle(scene.timeline[0])
        engine.handle(scene.timeline[1])
        early = engine.issue_switches("l1")
        # 未确定的开关不交给解码端
        self.assertEqual(early.flags[0, 1, 0], FLAG_UNDETERMINED)
        self.assertEqual(early.src_y[0, 1, 0], -1)
        self.assertTrue(np.isnan(early.values[0, 1, 0]))
        # 已确定的部分允许出具
        self.assertEqual(
            (int(early.src_y[1, 1, 0]), int(early.src_x[1, 1, 0]), float(early.values[1, 1, 0])),
            (1, 0, 4.0),
        )
        for event in scene.timeline[2:]:
            engine.handle(event)
        late = engine.issue_switches("l1")
        # 旧版继续按原视场解释：不被改写
        self.assertEqual(early.flags[0, 1, 0], FLAG_UNDETERMINED)
        # 新版补齐未确定项；两版共有的确定项完全一致
        self.assertEqual(late.flags[0, 1, 0], FLAG_DETERMINED)
        for cell in [(0, 0), (1, 0), (1, 1)]:
            np.testing.assert_array_equal(early.flags[cell], late.flags[cell])
            self.assertEqual(
                (int(early.src_y[cell + (0,)]), int(early.src_x[cell + (0,)])),
                (int(late.src_y[cell + (0,)]), int(late.src_x[cell + (0,)])),
            )


class SwitchFileTests(unittest.TestCase):
    def make_issues(self):
        scene = load_sample("scene.json")
        engine = Engine.from_scene(scene)
        run_tiles(engine, scene)
        return scene, engine, {l: engine.issue_switches(l) for l in scene.layers}

    def test_roundtrip_and_range_query(self):
        scene, engine, issues = self.make_issues()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "l0.sw"
            write_switch_file(path, issues["l0"])
            sf = SwitchFile(path)
            self.assertEqual(sf.layer, "l0")
            self.assertEqual(sf.data_version, 0)
            entries = {(gy, gx): (flag, sy, sx)
                       for gy, gx, flag, sy, sx in sf.query(0, 2, 0, 3)}
            self.assertEqual(len(entries), 6)
            self.assertEqual(entries[(0, 0)], (FLAG_DETERMINED, 1, 0))
            self.assertEqual(entries[(1, 1)], (FLAG_DETERMINED, 2, 2))
            part = sf.query(1, 2, 1, 3)
            self.assertEqual({(gy, gx) for gy, gx, *_ in part}, {(1, 1), (1, 2)})
            # 潜在影响范围按公开开关格推回（含 origin 平移）
            self.assertEqual(sf.potential_influence(0, 1), (0, 2, 1, 4))

    def test_flags_distinguish_valid_zero_no_coverage_pending(self):
        data = {
            "layers": {"m": {"shape": [1, 2], "kernel": [1, 1], "stride": [1, 1],
                             "dilation": [1, 1], "padding": [0, 0], "grid": [1, 2],
                             "origin": [0, 0]}},
            "primary": {"m": {"left": [0, 1, 0, 1], "right": [0, 1, 1, 2]}},
            "features": {"m": {"left": [[0.0]], "right": [[None]]}},
            "pool_dy": {"m": [[0.0, 0.0]]},
            "region_requests": [],
            "timeline": [],
        }
        engine = Engine.from_scene(parse_scene(data))
        engine.deliver_tile("m", "left")
        issue = engine.issue_switches("m")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.sw"
            write_switch_file(path, issue)
            entries = {(gy, gx): (f, sy, sx)
                       for gy, gx, f, sy, sx in SwitchFile(path).query(0, 1, 0, 2)}
            # 有效零：源 (0,0) 是合法胜出位置
            self.assertEqual(entries[(0, 0)], (FLAG_DETERMINED, 0, 0))
            # 待确认：右片缺测，源坐标写 -1
            self.assertEqual(entries[(0, 1)], (FLAG_UNDETERMINED, -1, -1))
        # 无覆盖：右片送达但整位置不可观测
        engine.deliver_tile("m", "right")
        issue = engine.issue_switches("m")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m2.sw"
            write_switch_file(path, issue)
            entries = {(gy, gx): (f, sy, sx)
                       for gy, gx, f, sy, sx in SwitchFile(path).query(0, 1, 0, 2)}
            self.assertEqual(entries[(0, 1)], (FLAG_NO_COVERAGE, -1, -1))

    def test_range_query_reads_only_covering_blocks(self):
        spec = LayerSpec("big", (64, 64), (1, 1), (1, 1), (1, 1), (0, 0), (40, 40), (0, 0))
        gh, gw = 40, 40
        issue = SwitchIssue(
            1, "big", 0, spec,
            np.ones((gh, gw, 1), np.uint8),
            np.zeros((gh, gw, 1), np.int32),
            np.zeros((gh, gw, 1), np.int32),
            np.zeros((gh, gw, 1)),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "big.sw"
            write_switch_file(path, issue, block=(16, 16))
            sf = SwitchFile(path)
            self.assertEqual(len(sf.directory["blocks"]), 9)
            got = sf.query(0, 8, 0, 8)
            self.assertEqual(len(got), 64)
            self.assertEqual(sf.blocks_read, 1)  # 非整图查询不解全部块
            sf.query(0, 40, 0, 40)
            self.assertEqual(sf.blocks_read, 9)

    def test_edge_decodes_from_file_and_pooled_values(self):
        scene, engine, issues = self.make_issues()
        with tempfile.TemporaryDirectory() as tmp:
            providers, pooled = {}, {}
            for lname, issue in issues.items():
                path = Path(tmp) / f"{lname}.sw"
                write_switch_file(path, issue)
                providers[lname] = SwitchFile(path).as_provider()
                pooled[lname] = issue.values
            for rect in scene.region_requests:
                via_file = decode_region(rect, providers, pooled)
                via_engine = engine.decode(rect)
                self.assertTrue(via_file.complete)
                np.testing.assert_array_equal(via_file.values, via_engine.values)


class BackwardTests(unittest.TestCase):
    def test_tile_gradients_route_to_true_owner(self):
        scene = load_sample("scene.json")
        engine = Engine.from_scene(scene)
        run_tiles(engine, scene)

        tiles = engine.tile_gradients("l0")
        want_left = np.zeros((3, 2, 1))
        want_left[1, 0, 0] = 1.0  # (0,0) 的 1；(1,0) 的 0 也是有效贡献
        want_right = np.zeros((3, 3, 1))
        want_right[0, 2, 0] = 5.0
        want_right[1, 1, 0] = -2.0
        want_right[2, 0, 0] = 1.0
        want_right[2, 1, 0] = 2.0
        self.assertTrue(tiles["left"].deliverable and tiles["right"].deliverable)
        np.testing.assert_array_equal(tiles["left"].grad, want_left)
        np.testing.assert_array_equal(tiles["right"].grad, want_right)

        tiles = engine.tile_gradients("l1")
        want_left = np.zeros((3, 2, 1))
        want_left[1, 0, 0] = 3.0  # 1 + 2，多次覆盖的贡献求和
        want_left[1, 1, 0] = 0.0  # 3 + 0 + 1 - 4
        np.testing.assert_array_equal(tiles["left"].grad, want_left)
        np.testing.assert_array_equal(tiles["right"].grad, np.zeros((3, 2, 1)))

    def test_deliverability_waits_for_switches_and_dy(self):
        scene = load_sample("scene.json")
        engine = Engine.from_scene(scene)
        engine.handle(("tile", "l0", "left"))
        tiles = engine.tile_gradients("l0", dy=scene.pool_dy["l0"])
        self.assertFalse(tiles["left"].deliverable)
        self.assertIn(("switch", 0, 1, 0), tiles["left"].waiting)
        self.assertIn(("switch", 1, 1, 0), tiles["left"].waiting)

        engine.handle(("tile", "l0", "right"))
        # 开关齐了但反向贡献未送齐
        tiles = engine.tile_gradients("l0", dy={(0, 0): 1.0})
        self.assertFalse(tiles["left"].deliverable)
        self.assertIn(("dy", 1, 0, 0), tiles["left"].waiting)
        tiles = engine.tile_gradients("l0", dy=scene.pool_dy["l0"])
        self.assertTrue(tiles["left"].deliverable)

    def test_backward_region_sums_in_layer_and_cell_order_fp64(self):
        # 三个窗口的胜出位置相同：贡献按 (层号, 开关格) 顺序以 FP64 累加
        data = {
            "layers": {"l0": {"shape": [1, 3], "kernel": [1, 3], "stride": [1, 1],
                              "dilation": [1, 1], "padding": [0, 1], "grid": [1, 3],
                              "origin": [0, 0]}},
            "primary": {"l0": {"solo": [0, 1, 0, 3]}},
            "features": {"l0": {"solo": [[-5.0, 10.0, 0.0]]}},
            "pool_dy": {"l0": [[1e16, 1.0, -1e16]]},
            "region_requests": [],
            "timeline": [],
        }
        engine = Engine.from_scene(parse_scene(data))
        engine.deliver_tile("l0", "solo")
        report = engine.backward((0, 1, 0, 3))
        self.assertTrue(report.complete)
        # 顺序 (0,0)→(0,1)→(0,2)：((1e16 + 1) - 1e16) == 0；换个顺序才会得 1
        self.assertEqual(report.values[0, 1], 0.0)

    def test_gradient_registry_uses_issued_switch_version(self):
        scene = load_sample("scene.json")
        engine = Engine.from_scene(scene)
        engine.handle(("tile", "l0", "left"))
        early = engine.issue_switches("l0")
        registry = GradientRegistry()
        rect = (0, 3, 0, 5)
        report = engine.backward(rect, providers={"l0": early.provider()},
                                 pool_dy={"l0": scene.pool_dy["l0"]}, registry=registry)
        self.assertFalse(report.complete)
        self.assertEqual(set(registry.expected("l0", early.issue_id)),
                         {(0, 0, 0), (1, 0, 0)})  # 只登记确定且落区的贡献

        engine.handle(("tile", "l0", "right"))
        late = engine.issue_switches("l0")
        registry2 = GradientRegistry()
        report = engine.backward(rect, providers={"l0": late.provider()},
                                 pool_dy={"l0": scene.pool_dy["l0"]}, registry=registry2)
        self.assertTrue(report.complete)
        self.assertEqual(len(registry2.expected("l0", late.issue_id)), 6)


class BudgetTests(unittest.TestCase):
    def test_scene_timeline_within_budget(self):
        scene = load_sample("scene.json")
        engine = Engine.from_scene(scene)
        run_tiles(engine, scene)
        self.assertLess(engine.budget.total, scene.budget_bytes)

    def test_tight_budget_raises(self):
        scene = load_sample("scene.json")
        engine = Engine.from_scene(scene, budget_bytes=16)
        engine.deliver_tile("l0", "left")
        with self.assertRaises(BudgetExceeded):
            engine.evaluate_cell("l0", 0, 0)

    def test_full_expansion_would_exceed_budget(self):
        # 3072×4096 整层、2×2 核：完整窗口展开远超预算，必须按查询惰性求值
        expansion = 3072 * 4096 * 2 * 2 * Budget.CANDIDATE_BYTES
        self.assertGreater(expansion, 262144)


if __name__ == "__main__":
    unittest.main()
