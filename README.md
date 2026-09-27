# ellipse_filter — 纹理各向异性足迹的椭圆滤波

离线图形采样用可复用模块:对标量纹理和 RGBA 纹理做 EWA 风格的椭圆高斯
足迹滤波。
纯 Python 标准库实现,无第三方依赖。

## 环境

- Python 3.14.7,Linux 原生运行
- 仅标准库,安装 Python 后无需任何额外依赖(`requirements.txt` 为空注释)

## 命令

```bash
python -m unittest discover -s tests -v   # 运行全部测试
python demo.py                            # 运行演示(约 1 秒内完成)
```

## 接口

```python
from ellipse_filter import AlphaMode, WrapMode, filter_sample

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

# RGBA:每个像素是 4 通道序列 (r, g, b, a),必须显式声明 alpha 表示
result = filter_sample(
    rgba_texture,
    u,
    v,
    jacobian,
    alpha_mode=AlphaMode.STRAIGHT,       # 或 AlphaMode.PREMULTIPLIED
)
# result.value == (r, g, b, a)
```

单通道旧格式仍为 `texture[j][i] = float`;RGBA 格式为
`texture[j][i] = (r, g, b, a)`。两种格式不能在同一纹理中混用。RGBA 的
`alpha_mode` 必须显式提供;`AlphaMode.UNPREMULTIPLIED` 是
`AlphaMode.STRAIGHT` 的同义成员。

## 数学模型

```
C = J * J^T + 0.25 * I                    (对任意有限 J 均对称正定)
足迹: d^T * C^-1 * d <= 4                 (d 为像素中心相对 (u,v) 的偏移)
核权: w = exp(-2 * d^T * C^-1 * d)
输出: value = sum(w_k * t_k) / sum(w_k)
```

RGBA 四个通道使用完全相同的空间权重 `w_k`,alpha 不会再乘一次颜色权重。
为避免完全透明像素中无意义的 RGB 在边缘形成色边,滤波统一在预乘空间进行:

```
straight 输入: (r, g, b, a) -> (r*a, g*a, b*a, a);a=0 时 RGB 视为 0
预乘累加:     (R, G, B, A) = sum(w_k * premul_t_k) / sum(w_k)
straight 输出: (R/A, G/A, B/A, A);A=0 时输出 (0, 0, 0, 0)
```

因此 `(0,0,0,0)` 表示完全透明,`(0,0,0,1)` 仍表示真正的不透光黑色。
`AlphaMode.PREMULTIPLIED` 的输入和输出都保持预乘表示;alpha 为 0 的输入
像素按预乘约定视为黑色,不泄漏其存储颜色。

遍历足迹的轴对齐包围盒(半宽 `2*sqrt(C_00)`、`2*sqrt(C_11)`)内的全部
像素中心,对满足足迹条件者累加核权。`repeat` 取模回绕、`clamp` 夹到
边缘;多个候选中心映射到同一存储像素时,**每个副本保留自己的核权**,
独立参与加权。

## 输入上限与错误语义

- 纹理最大 64x64,至少 1x1,必须矩形;标量像素为数字,RGBA 像素必须恰好
  4 个数值通道,alpha 必须位于 `[0, 1]`;`budget` 默认 4096。
- `InvalidInputError`(同时是 `ValueError`):数值输入为 NaN/Infinity、
  bool 充当数值(含 `budget`)、纹理非矩形/为空/超限、Jacobian 非 2x2、
  `wrap` 非 `WrapMode` 成员、RGBA 未提供 `AlphaMode`、`alpha_mode` 非法、
  标量/RGBA 格式混用、`budget` 非正整数。错误信息指明出错的参数(如
  `texture[2][5][1]`、`jacobian[0][1]`)。
- `BudgetExceededError`(同时是 `RuntimeError`,与 `InvalidInputError`
  互不继承):包围盒候选数超过 budget,或 `C` 因巨大 J 溢出为非有限值。
  携带 `candidates` 与 `budget` 属性。**不静默退化为点采样**,异常前不
  产生任何部分结果;预算耗尽与"足迹为空"是不同路径(后者对任意有限
  输入在数学上不可达:最近像素中心的 q <= 2 < 4)。

## 误差口径(浮点)

- 常量纹理守恒:`|value - c| <= 1e-12`(仅含浮点累加舍入)。
- 归一化权重和:`|sum(weights) - 1| <= 1e-12`。
- 单通道与独立暴力宽盒参考(测试内独立实现,扫描加大盒、独立求逆与累加)
  比较:值与权重和绝对差 `<= 1e-9`,贡献数严格相等;RGBA 的权重与相同
  空间足迹下的单通道结果一致。
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
- **统一预乘空间滤波**:透明像素的 RGB 不参与颜色贡献,从根上避免透明
  存储色在滤波边缘形成色边;输出再按调用方声明转回未预乘表示。
- **不可变结果**:`FilterResult` 为 frozen dataclass,权重以元组返回。

## 已知限制

- 仅支持标量和 RGBA 纹理;不实现其他通道排列、mip 链、三线性或饱和截断
  (summed-area)。
- 重复/夹取只按本题两种模式实现,不宣称兼容任何图形 API 的完整寻址
  模式集(如 mirror、border)。
- 足迹核在 q=4 处硬截断(边界权约 e^-8 ≈ 3.35e-4),不做尾部补偿。
- 纹理尺寸上限 64x64、budget 上限语义按本题契约,非行业标准实现。

## 文件

- `ellipse_filter.py` — 核心库(可导入)
- `tests/test_ellipse_filter.py` — 单元测试与独立暴力宽盒参考
- `demo.py` — 固定输入演示:点采样 vs 足迹滤波 + budget/非法输入拒绝
