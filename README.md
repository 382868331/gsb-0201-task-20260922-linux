# ewa_filter — 纹理各向异性足迹的椭圆滤波

可复用的椭圆足迹滤波（EWA 风格）模块，用于离线图形采样。**Python 3.14.7，仅标准库，无任何第三方依赖**，Linux 原生运行，核心、测试、演示全部离线。

## 模型

给定标量纹理、采样坐标 `(u, v)` 与 2×2 屏幕到纹理 Jacobian `J`：

- 滤波协方差 `C = J·Jᵀ + 0.25·I`（0.25I 为 0.5 texel 的基础滤波，保证 C 恒正定，包括退化 J）
- 足迹椭圆 `q(d) = dᵀ·C⁻¹·d ≤ 4`，在纹素中心 `d = (i+0.5−u, j+0.5−v)` 处求值
- 每个落入足迹的纹素中心以高斯权 `w = exp(−2·q)` 贡献，输出归一化加权平均 `Σwₖtₖ / Σwₖ`

坐标约定：纹理坐标以 texel 为单位，纹素 `(i, j)` 覆盖 `[i, i+1)×[j, j+1)`，中心 `(i+0.5, j+0.5)`；`u` 对应列（x），`v` 对应行（y）；Jacobian 把屏幕像素映射到该 texel 单位。

## 接口

```python
from ewa_filter import filter_footprint, FilterResult, Contribution

res = filter_footprint(
    texture,        # 行的序列，texture[y][x]，1..64 × 1..64，有限实数
    u, v,           # 采样坐标（texel 单位，有限实数）
    jacobian,       # [[du/dx, du/dy], [dv/dx, dv/dy]]，有限实数
    wrap="repeat",  # "repeat"（取模）或 "clamp"（夹边）
    max_candidates=4096,  # 包围盒候选预算，int >= 1
)
# res.value          归一化滤波值
# res.weight_sum     原始核权之和
# res.count          实际贡献数
# res.contributions  tuple[Contribution(texel_x, texel_y, weight)]，weight 为未归一化核权
```

- 候选枚举走足迹的轴对齐包围盒（半轴 `2√C₁₁`、`2√C₂₂`），再按 `q ≤ 4` 精确筛选。
- **repeat/clamp 下同一纹素被多个候选位置映射到时，每个副本保留自己的核权**，不合并——`contributions` 中同一 `(texel_x, texel_y)` 可出现多次，`count` 计入全部副本。
- 包围盒候选数超过 `max_candidates`（默认 4096）时，在采样前抛 `BudgetExceededError`（携带 `candidates`/`limit` 属性），**绝不静默退化为点采样**，也不产生部分结果。

## 输入上限

- 纹理宽高均 ≤ 64，非空、非锯齿（每行等长）
- 候选预算默认 4096，可经由 `max_candidates` 调整（int ≥ 1）

## 错误语义

所有异常继承 `ewa_filter.FootprintError`：

- `InvalidInputError`（也是 `ValueError`）：参数违反契约。数值输入拒绝 NaN/±Infinity；整数参数（`max_candidates`）拒绝 bool 与非整型；纹理拒绝空、锯齿、超 64²、非有限值；Jacobian 必须 2×2；`wrap` 只接受 `"repeat"`/`"clamp"`。错误消息指明出错参数，不静默纠正。
- `BudgetExceededError`：输入合法但搜索预算耗尽。与输入非法是**不同的错误类型**，调用方可区分“预算耗尽”与“契约违反”。

函数为纯函数：任何失败都不会留下部分变更。

## 误差口径

全部计算使用 IEEE 754 双精度。`value` 是贡献纹素的加权平均：

- 常量纹理守恒到 ~1e-15 相对量级（测试阈值 1e-12，非宽松容差）
- 与独立暴力宽盒积分参考的 `value`/`weight_sum` 偏差 < 1e-12（相对），`count` 完全相等
- 足迹边界 `q = 4` 严格按 `q ≤ 4.0` 判定；边界附近的纹素归属是舍入敏感的，测试用精确可表示的参数（`J = 0.5I`、`J = 0`）验证边界行为，不用容差掩盖

## 设计取舍与已知限制

- 仅标量纹理；多通道需逐通道调用
- 候选枚举用轴对齐包围盒（对强旋转椭圆偏保守），预算按包围盒计而非按最终贡献数计
- 无 mipmap/多级滤波：超出预算直接报错，由调用方决定降采样策略
- 只实现本题契约的数学子集（`C = JJᵀ + 0.25I`、`q ≤ 4`、`exp(−2q)`、repeat/clamp），不宣称兼容任何完整行业纹理过滤标准

## 运行

```bash
python -m unittest discover -s tests -v   # 26 个测试
python demo.py                            # 点采样 vs 足迹滤波对比 + 真实拒绝演示，约 0.1s
```

测试中的暴力宽盒积分参考在 `tests/test_ewa_filter.py` 内独立重写（加宽盒、独立循环），不调用被测核心。demo 数据由确定性代码生成（正弦渐变 + 棋盘格），无任何外部输入。
