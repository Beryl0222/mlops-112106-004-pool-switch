# 瓦片池化与开关交接

samples/scene.json 是本地合成的特征层和区域读者。可检查资料的形状、来源关系或交换文件边界。时间序列条目表示调用方送达或发起读取的顺序，数值为输入。

完整交换约定见 [交换格式](exchange.md)。运行环境为 Python 3.11 和 NumPy 2.2.6，使用本地文件即可运行资料工具。

```sh
python3 -m pip install -r requirements.txt
python3 inspect_exchange.py samples/scene.json
python3 -m unittest discover -s tests -v
```

## 张量路径库

`poolswitch/` 把池化、开关交接与局部上采样做成边端与训练共用的 Python 张量路径：

- `geometry`：窗口在整层坐标上定义；定位重叠窗口只做算术反解，不把完整跨层展开张量放进工作区。
- `scene`：解析交换资料；原始图、有效视场（所有权契约）与特征值分开提供。
- `engine`：流式最大池化。图外补齐按负无穷、缺测与已确认不可观测分开、真值 0 合法；平局先按全层 (y, x) 再按通道。开关一律以整层坐标出具；缺片仍可能改变胜出位置的开关不交给解码端，可信上界持平或更低时仍允许确定已定的部分。
- `region`：区域解码与梯度回填的共用路径，接受活引擎、开关快照或二进制文件三种单元提供者；同点入射贡献按层号、开关格、通道顺序以 FP64 累加，区域完整性与浮点输出分开报告。
- `switchfile`：公开二进制开关格式，块目录支持范围读取，非整图查询不必解出全部开关；分清有效零、无覆盖、待确认。
- `reference`：全图稠密参照；换瓦片划分、换请求区域后仍能对照整层定义检验同一激活和梯度落在何处。

```python
from poolswitch import Engine, load_scene

scene = load_scene("samples/scene.json")
engine = Engine.from_scene(scene)          # 预算按 budget_bytes 记账
for event in scene.timeline:
    report = engine.handle(event)          # 瓦片送达 / 区域请求
issue = engine.issue_switches("l0")        # 出具整层坐标开关快照
tiles = engine.tile_gradients("l0")        # 反向贡献归回真正拥有者
```
