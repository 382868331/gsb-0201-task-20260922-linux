# ellipse_filter — 纹理各向异性足迹的椭圆滤波

离线图形采样用可复用模块:对标量纹理做 EWA 风格的椭圆高斯足迹滤波。
纯 Python 标准库实现,无第三方依赖。

支持**单纹理**与**显式瓦片图册(tile atlas)**两种入口:每个瓦片有显式
ID 和各自尺寸,可配置每瓦片的 clamp/repeat 边界模式;滤波足迹越界时只
按选定模式在**该瓦片自身**的网格内取样,不会串入其他瓦片。

## 环境

- Python 3.14.7,Linux 原生运行
- 仅标准库,安装 Python 后无需任何额外依赖(`requirements.txt` 为空注释)

## 命令

```bash
python -m unittest discover -s tests -v   # 运行全部测试
python demo.py                            # 运行演示(约 1 秒内完成)
```

## 接口

### 单纹理(旧入口,行为不变)

```python
from ellipse_filter import filter_sample, WrapMode, FilterResult

result = filter_sample(
    texture,      # 序列的序列:texture[j][i],行=v 方向,列=u 方向
    u, v,         # 纹理空间采样坐标(像素中心位于整数+0.5)
    jacobian,     # 2x2 屏幕到纹理 Jacobian J = ((du/dx, du/dy), (dv/dx, dv/dy))
    wrap=WrapMode.REPEAT,   # 或 WrapMode.CLAMP
    budget=4096,            # 包围盒候选像素上限
)
# result.value       归一化滤波值
# result.weights     每个贡献的归一化核权(和为 1),按行主序扫描顺序
# result.texels      与 weights 对齐的 (i, j) 像素索引;同一像素可重复出现
# result.count       实际贡献数
# result.weight_sum  未归一化的核权总和
# result.tile_id     单纹理入口恒为 None
```

### 瓦片图册(新入口)

```python
from ellipse_filter import TileAtlas, filter_tile, WrapMode

# 三种等价构造:
atlas = TileAtlas({"albedo": tex0, "normal": tex1})          # dict: id -> 纹理
atlas = TileAtlas([("albedo", tex0), ("normal", tex1)])      # 序列: (id, 纹理)
atlas = TileAtlas({"a": (tex0, WrapMode.CLAMP)})             # dict 值可为 (纹理, wrap)
atlas = TileAtlas([("a", tex0, WrapMode.CLAMP)])             # 三元组 (id, 纹理, wrap)
atlas = TileAtlas({0: tex0, 7: tex1})                        # id 可为任意可哈希非布尔值

atlas = TileAtlas(tiles, default_wrap=WrapMode.CLAMP)        # 图册级缺省边界模式

atlas.tile_ids          # 构造顺序的全部 id
atlas.size("albedo")    # (width, height)
len(atlas); "albedo" in atlas

result = filter_tile(
    atlas, "albedo", u, v, jacobian,
    wrap=None,          # None=用瓦片自身配置;显式传入则本次调用强制覆盖
    budget=4096,
)
# result.tile_id 回显所采样的瓦片 id;texels 是该瓦片内部索引
```

**边界模式解析优先级**(仅当 `wrap=None`):本次显式 `wrap` > 瓦片自身
wrap > 图册 `default_wrap` > 全局缺省 `REPEAT`。

**隔离保证**:足迹只在 `tile_id` 指定瓦片的纹理空间内展开;越界候选中心
只按该瓦片自身的宽/高取模或夹边,结果中不可能出现其他瓦片的像素,无论
`u/v` 离瓦片多远。瓦片尺寸彼此独立(如 3×2、7×1、1×5、16×9 可共存)。

`TileAtlas` 构造时即校验全部瓦片,任一瓦片非法则整个图册构造失败
(不会留下半填充图册);对象不可变(`MappingProxyType` + 禁止属性赋值)。

## 数学模型

```
C = J * J^T + 0.25 * I                    (对任意有限 J 均对称正定)
足迹: d^T * C^-1 * d <= 4                 (d 为像素中心相对 (u,v) 的偏移)
核权: w = exp(-2 * d^T * C^-1 * d)
输出: value = sum(w_k * t_k) / sum(w_k)
```

遍历足迹的轴对齐包围盒(半宽 `2*sqrt(C_00)`、`2*sqrt(C_11)`)内的全部
像素中心,对满足足迹条件者累加核权。`repeat` 取模回绕、`clamp` 夹到
边缘;多个候选中心映射到同一存储像素时,**每个副本保留自己的核权**,
独立参与加权。

## 输入上限与错误语义

- 纹理/瓦片最大 64x64,至少 1x1,必须矩形;`budget` 默认 4096。
- `InvalidInputError`(同时是 `ValueError`):数值输入为 NaN/Infinity、
  bool 充当数值(含 `budget` 与瓦片 id)、纹理非矩形/为空/超限、
  Jacobian 非 2x2、`wrap` 非 `WrapMode` 成员、`budget` 非正整数、
  图册规格格式错误/瓦片 id 重复或不可哈希/瓦片 id 为 `None`、
  `filter_tile` 的 `atlas` 不是 `TileAtlas`。错误信息指明出错的参数
  (如 `texture[2][5]`、`jacobian[0][1]`、`tile_id`)。
- `TileNotFoundError`(`InvalidInputError` 的子类,故也是
  `ValueError`):`filter_tile` 给定的 id 不在图册中。携带 `tile_id` 与
  `available`(图册中实际存在的 id 元组),便于定位。
- `BudgetExceededError`(同时是 `RuntimeError`,与 `InvalidInputError`
  互不继承):包围盒候选数超过 budget,或 `C` 因巨大 J 溢出为非有限值。
  携带 `candidates` 与 `budget` 属性。**不静默退化为点采样**,异常前不
  产生任何部分结果;预算耗尽与"足迹为空"是不同路径(后者对任意有限
  输入在数学上不可达:最近像素中心的 q <= 2 < 4)。
- 图册构造为**全有或全无**:逐瓦片校验,任一失败即抛错,不存在半更新、
  半填充的图册;构造后的 `TileAtlas` 不可变。

## 误差口径(浮点)

- 常量纹理守恒:`|value - c| <= 1e-12`(仅含浮点累加舍入)。
- 归一化权重和:`|sum(weights) - 1| <= 1e-12`。
- 与独立暴力宽盒参考(测试内独立实现,扫描加大盒、独立求逆与累加)
  比较:值与权重和绝对差 `<= 1e-9`,贡献数严格相等。
- 脉冲解析权重:与闭式 `exp(-8)` 结果绝对差 `<= 1e-15`。
- 结果本身是离散像素中心加权和,**不是**连续 EWA 积分;与连续积分的
  偏差随足迹增大而减小,本模块不声称二者等价。

## 设计取舍

- **像素中心离散求和**而非连续积分:契约即"遍历包围盒像素中心",
  实现简单、可复现、与暴力参考可逐位对齐。
- **先算候选数再遍历**:巨大 J 时包围盒可达天文尺寸,先用整数区间
  长度相乘得到候选数并检查 budget,避免在报错前陷入长时间循环。
- **0.25I 正则项**:使退化/零 Jacobian 仍有定义(退化为半径 1 的
  各向同性足迹),不拒绝合法输入。
- **不可变结果**:`FilterResult` 为 frozen dataclass,权重以元组返回;
  新增字段 `tile_id` 带缺省值 `None`,旧位置/关键字构造与断言不受影响。
- **图册即不相交存储的集合**:`filter_tile` 与 `filter_sample` 共用同
  一份足迹核心 `_run_filter`,差别只在取哪张已校验的二维数组及用哪组
  宽/高做回绕,因此瓦片入口与单纹理入口在同一网格上逐位一致(有测试)。
- **图册不可变且全有或全无构造**:`MappingProxyType` 封装内部映射、禁止
  属性赋值;逐瓦片 eager 校验,杜绝半更新状态。
- **`TileNotFoundError` 归入 `InvalidInputError`**:未知 id 属于"参数
  不合法",但单独成类以便调用方专门捕获;它与 budget 耗尽路径互斥。

## 已知限制

- 仅标量(单通道)纹理;不实现 mip 链、三线性或饱和截断(summed-area)。
- 重复/夹取只按本题两种模式实现,不宣称兼容任何图形 API 的完整寻址
  模式集(如 mirror、border)。
- 足迹核在 q=4 处硬截断(边界权约 e^-8 ≈ 3.35e-4),不做尾部补偿。
- 纹理尺寸上限 64x64、budget 上限语义按本题契约,非行业标准实现。
- 瓦片按各自独立的纹理空间寻址;本次**不**做跨面片导数拆分或图册缝线
  重参数化(seam reparameterization),相邻瓦片在几何上不连续。
- 图册本身不按瓦片数设上限,但每个瓦片仍受 64x64 限制。

## 文件

- `ellipse_filter.py` — 核心库(可导入):`filter_sample`、`filter_tile`、
  `TileAtlas`、`WrapMode`、错误类型
- `tests/test_ellipse_filter.py` — 旧单元测试与独立暴力宽盒参考(未改)
- `tests/test_tiles.py` — 瓦片图册测试:角点/负 UV/不同尺寸/隔离/每瓦片
  wrap/构造与失败路径,含独立宽盒参考与手算十字足迹
- `demo.py` — 固定输入演示:点采样 vs 足迹滤波 + budget/非法输入拒绝
  + 瓦片图册(边界模式、隔离、TileNotFoundError)
