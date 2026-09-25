# ellipse_filter — 纹理各向异性足迹的椭圆滤波

离线图形采样用可复用模块:对标量纹理做 EWA 风格的椭圆高斯足迹滤波。
纯 Python 标准库实现,无第三方依赖。

## 环境

- Python 3.14.7,Linux 原生运行
- 仅标准库,安装 Python 后无需任何额外依赖(`requirements.txt` 为空注释)

## 命令

```bash
./.venv/bin/python -m unittest discover -s tests -v   # 运行全部测试(新旧回归)
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

### 对采样中心 (u, v) 的解析导数

    from ellipse_filter import filter_sample_grad, GradientResult

    g = filter_sample_grad(
        texture, u, v, jacobian,
        wrap=WrapMode.REPEAT,
        budget=4096,        # 与 filter_sample 共用同一预算统计,
                            # 导数模式不会绕过 4096 上限
        boundary_tol=1e-9,  # q 距足迹边界 4.0 的判定容差
        edge_tol=1e-9,      # 包围盒边缘距 texel 中心的判定容差
    )
    # g.value          与 filter_sample 同输入完全一致(同一扫描与累加)
    # g.grad_u/g.grad_v 固定 Jacobian 下 value 对 u/v 的解析导数
    # g.differentiable 是否位于可微位置
    # g.nondifferentiable_reason  不可微原因(可微时为 None)
    # g.weights/texels/count/weight_sum  与 FilterResult 相同契约

导数按解析链式法则计算:对 value = A/B(A = sum(w_k*t_k),B = sum(w_k),
w_k = exp(-2*q_k))同时微分权重与归一化分母(商法则),
dq/du = -2*(inv00*dx + inv01*dy),dw/du = -2*w*dq/du。
repeat/clamp 把多个候选中心映射到同一存储 texel 时,每个副本各自
携带自己的核权导数参与求和,而不是对最终像素值做差分。

以下位置值函数不可微,结果会被显式标记(differentiable=False
并给出原因),梯度字段仍会填出但不可信:

- 候选距离阈值附近:包围盒内某候选的 q 与足迹边界 4.0 之差不超过
  boundary_tol(核在 q=4 处硬截断,候选进出时权值跳动 e^-8);
- clamp/repeat 映射切换处:包围盒边缘与 texel 中心的距离不超过
  edge_tol,此时候选集合(以及 repeat/clamp 下同一存储 texel 的副本
  重数)在无穷小扰动下离散变化。

其余位置(普通安全位置)返回解析导数,可用中心差分核验。


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

- 纹理最大 64x64,至少 1x1,必须矩形;`budget` 默认 4096。
- `InvalidInputError`(同时是 `ValueError`):数值输入为 NaN/Infinity、
  bool 充当数值(含 `budget`)、纹理非矩形/为空/超限、Jacobian 非 2x2、
  `wrap` 非 `WrapMode` 成员、`budget` 非正整数。错误信息指明出错的参数
  (如 `texture[2][5]`、`jacobian[0][1]`)。
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
- 解析梯度:与测试内独立实现的商法则参考绝对差 <= 1e-9;与
  filter_sample 的中心差分(h=1e-5)在平滑纹理、安全位置绝对差
  <= 1e-6;常量纹理梯度 |grad| <= 1e-12(仅累加舍入)。
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
- 由于硬截断,值函数在候选跨越 q=4 处存在约 e^-8 量级的跳变,
  导数在这些点不存在;filter_sample_grad 只做检测与标记,
  不提供跨跳变的次梯度或平滑化。
- 纹理尺寸上限 64x64、budget 上限语义按本题契约,非行业标准实现。

## 文件

- `ellipse_filter.py` — 核心库(可导入)
- `tests/test_ellipse_filter.py` — 单元测试与独立暴力宽盒参考
- `tests/test_ellipse_grad.py` — 导数测试:独立商法则参考、中心差分核验、
  repeat 缝两侧、clamp 边外、重合副本、不可微标记、巨大足迹预算
- `demo.py` — 固定输入演示:点采样 vs 足迹滤波 + budget/非法输入拒绝 +
  解析导数与中心差分对照
