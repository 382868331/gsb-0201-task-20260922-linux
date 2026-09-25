# ellipse_filter — 纹理各向异性足迹的椭圆滤波

离线图形采样用可复用模块:对标量纹理做 EWA 风格的椭圆高斯足迹滤波。
纯 Python 标准库实现,无第三方依赖。

## 环境

- Python 3.14.7,Linux 原生运行
- 仅标准库,安装 Python 后无需任何额外依赖(`requirements.txt` 为空注释)

## 命令

```bash
./.venv/bin/python -m unittest discover -s tests -v   # 运行全部测试
./.venv/bin/python demo.py                            # 运行演示(约 1 秒内完成)
```

## 接口

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
```

固定 Jacobian 下对采样坐标的解析导数:

```python
from ellipse_filter import filter_sample_with_derivatives, GradientResult

grad = filter_sample_with_derivatives(
    texture, u, v, jacobian,
    wrap=WrapMode.REPEAT,
    budget=4096,                     # 与 filter_sample 同一预算,求导不放宽
    boundary_tol=1e-9,               # 不可微检测容差(默认 DEFAULT_BOUNDARY_TOL)
)
# grad.sample            FilterResult,与 filter_sample 同参数结果逐位一致
# grad.dvalue_du         value 对 u 的解析偏导(Jacobian 固定)
# grad.dvalue_dv         value 对 v 的解析偏导
# grad.nondifferentiable 位于不可微轨迹 boundary_tol 邻域内时为 True
# grad.reasons           每条附近不可微轨迹的文字说明(安全位置为空)
```

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

## 导数模型

固定 Jacobian,对采样中心 `(u, v)` 求解析导数。权重与归一化分母都
参与求导(商法则):

```
value = N / D,  N = sum(w_k * t_k),  D = sum(w_k)
dw_k/du = -2 * w_k * dq_k/du,  dq_k/du = -2 * (C^-1 d)_u
dvalue/du = (dN/du * D - N * dD/du) / D^2        (v 同理)
```

重复/夹取映射到同一存储像素的副本各自算核权与核权导数后再汇总;
导数不是对最终像素值的差分。

候选集合固定时 value 关于 `(u, v)` 光滑;当某像素中心恰好落在足迹
边界 `q = 4` 上时候选集合跳变(权重跳变 e^-8),若该中心在纹理范围
外,其 repeat/clamp 映射同一时刻切换。`filter_sample_with_derivatives`
显式探测包围盒外扩一格内的全部像素中心,`|q - 4| <= boundary_tol`
即置 `nondifferentiable=True` 并在 `reasons` 中说明(含映射切换信息);
此时返回的导数是当前候选配置下的单侧解析值,仅供参考。普通安全
位置返回解析导数,可用中心差分核验。

## 输入上限与错误语义

- 纹理最大 64x64,至少 1x1,必须矩形;`budget` 默认 4096。
- `InvalidInputError`(同时是 `ValueError`):数值输入为 NaN/Infinity、
  bool 充当数值(含 `budget`)、纹理非矩形/为空/超限、Jacobian 非 2x2、
  `wrap` 非 `WrapMode` 成员、`budget` 非正整数。错误信息指明出错的参数
  (如 `texture[2][5]`、`jacobian[0][1]`)。
   导数接口额外校验 `boundary_tol` 为有限非负浮点数。
- `BudgetExceededError`(同时是 `RuntimeError`,与 `InvalidInputError`
  互不继承):包围盒候选数超过 budget,或 `C` 因巨大 J 溢出为非有限值。
  携带 `candidates` 与 `budget` 属性。**不静默退化为点采样**,异常前不
  产生任何部分结果;预算耗尽与"足迹为空"是不同路径(后者对任意有限
  输入在数学上不可达:最近像素中心的 q <= 2 < 4)。

## 误差口径(浮点)

- 常量纹理守恒:`|value - c| <= 1e-12`(仅含浮点累加舍入)。
- 归一化权重和:`|sum(weights) - 1| <= 1e-12`。
- 与独立暴力宽盒参考(测试内独立实现,扫描加大盒、独立求逆与累加)
  比较:值与权重和绝对差 `<= 1e-9`,贡献数严格相等。
- 脉冲解析权重:与闭式 `exp(-8)` 结果绝对差 `<= 1e-15`。
 - 解析导数:在安全位置与中心差分(h=1e-6)绝对差 `<= 1e-6`;常量纹理
   梯度 `<= 1e-12`(商法则下分子分母导数相消,仅剩舍入)。
- 结果本身是离散像素中心加权和,**不是**连续 EWA 积分;与连续积分的
  偏差随足迹增大而减小,本模块不声称二者等价。

## 设计取舍

- **像素中心离散求和**而非连续积分:契约即"遍历包围盒像素中心",
  实现简单、可复现、与暴力参考可逐位对齐。
- **先算候选数再遍历**:巨大 J 时包围盒可达天文尺寸,先用整数区间
  长度相乘得到候选数并检查 budget,避免在报错前陷入长时间循环。
- **0.25I 正则项**:使退化/零 Jacobian 仍有定义(退化为半径 1 的
  各向同性足迹),不拒绝合法输入。
- **不可变结果**:`FilterResult` 为 frozen dataclass,权重以元组返回。

## 已知限制

- 仅标量(单通道)纹理;不实现 mip 链、三线性或饱和截断(summed-area)。
- 重复/夹取只按本题两种模式实现,不宣称兼容任何图形 API 的完整寻址
  模式集(如 mirror、border)。
- 足迹核在 q=4 处硬截断(边界权约 e^-8 ≈ 3.35e-4),不做尾部补偿。
- 纹理尺寸上限 64x64、budget 上限语义按本题契约,非行业标准实现。
 - 导数仅对 `(u, v)` 定义(Jacobian 固定);不提供对 J 或纹理值的导数。
   `nondifferentiable=True` 时导数为单侧参考值;`boundary_tol` 只控制
   检测邻域,不改变数值结果。
 - 不加入瓦片(tiling)缓存或多级纹理(mipmap)。

## 文件

- `ellipse_filter.py` — 核心库(可导入)
- `tests/test_ellipse_filter.py` — 单元测试与独立暴力宽盒参考
- `demo.py` — 固定输入演示:点采样 vs 足迹滤波 + budget/非法输入拒绝 +
  解析导数对比中心差分 + 不可微轨迹标记
- `tests/test_ellipse_filter_gradients.py` — 导数测试:中心差分核验、
  边界/映射切换检测、预算与校验契约
