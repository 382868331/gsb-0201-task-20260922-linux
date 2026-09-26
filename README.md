# ellipse_filter + ripmap — 纹理各向异性足迹滤波

离线图形采样用可复用模块,包含两部分:

- ellipse_filter.py:对标量纹理做 EWA 风格的椭圆高斯足迹滤波(基线,接口不变)。
- ripmap.py:ripmap 构建与采样,沿 u/v 两个轴独立选择分辨率,
  不把长条足迹压成一个标量 LOD。

纯 Python 标准库实现,无第三方依赖。

## 环境

- Python 3.14.7,Linux 原生运行
- 仅标准库,安装 Python 后无需任何额外依赖(requirements.txt 为空注释)

## venv 准备

    python3.14 -m venv .venv        # 或 python -m venv .venv(确认版本为 3.14.7)
    source .venv/bin/activate
    # 无第三方依赖,不需要 pip install

## 命令

    python -m unittest discover -s tests -v   # 运行全部测试
    python demo.py                            # EWA 椭圆滤波演示(约 1 秒内完成)
    python demo_iteration.py                  # ripmap 演示(现场计算,远小于 90 秒)

## ripmap 接口

    from ripmap import build_ripmap, ripmap_sample, Ripmap, RipmapResult
    from ellipse_filter import WrapMode

    ripmap = build_ripmap(texture)   # texture[j][i],最大 64x64,契约同 EWA 入口
    result = ripmap_sample(
        ripmap,
        u, v,         # 纹理空间采样坐标(像素中心位于整数+0.5)
        jacobian,     # 2x2 屏幕到纹理 Jacobian
        wrap=WrapMode.REPEAT,   # 或 WrapMode.CLAMP,层内寻址约定与 EWA 一致
    )
    # result.value      四角层级混合后的最终值
    # result.level_u    连续(小数)层级,按 sqrt(C_00) 取 log2 后夹到已建层级
    # result.level_v    同上,按 sqrt(C_11)
    # result.corners    四个 LevelSample:整数层级、该层尺寸、层级混合权、
    #                   层内双线性采样到的像素索引与权重(权重和为 1)
    # result.count      全部角点的像素贡献总数

### ripmap 数学模型

- 层级 (lu, lv):基础纹理沿 u 降采样 2^lu 倍、沿 v 降采样 2^lv 倍;
  每步对 2 个相邻像素做面积平均。奇数边长时末尾输出像素只覆盖 1 个
  输入像素,按实际覆盖权重(2 或 1)平均,不丢最后一列/行。
  所有层尺寸通过 ripmap.all_sizes() / ripmap.size(lu, lv) 查询。
- 层级选择:C = J * J^T + 0.25 * I(与 EWA 同一足迹模型),
  lambda_u = log2(sqrt(C_00))、lambda_v = log2(sqrt(C_11)),
  夹到 [0, levels-1];在四个相邻整数层级间按双线性权混合。
- 层内:双线性采样,repeat 取模回绕、clamp 夹到边缘,与 EWA 入口一致。
- 退化足迹(零/奇异 Jacobian)合法,层级夹到 0;J 溢出为非有限值时
  对应轴取最粗层级。非法输入(NaN/Inf、bool、畸形 Jacobian/纹理、
  非法 wrap)抛 InvalidInputError。

### ripmap 误差口径

- 常量纹理守恒:|value - c| <= 1e-12。
- 角点层级混合权之和、每个角点层内权重之和均为 1(<= 1e-12 误差)。
- 与测试内独立参考(对原图做足迹盒 [u-su, u+su] x [v-sv, v+sv] 的
  精确面积积分,逐像素覆盖裁剪,repeat 寻址)比较:
  水平高频/竖向平滑纹理在 su=8 时误差 < 0.06;各向同性 su=sv=2 的
  平滑纹理误差 < 0.15(ripmap 的帐篷形预滤波只是盒式积分的近似)。
- ripmap 是轴对齐盒近似:旋转椭圆用其轴对齐外接范围近似,
  **不是**任意旋转椭圆足迹的精确积分;误差随旋转角与偏心率增大而增大。

## EWA 接口(基线,不变)

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

## 文件

- `ellipse_filter.py` — EWA 椭圆足迹滤波核心库(可导入,接口不变)
- `ripmap.py` — ripmap 构建与采样(可导入,复用 ellipse_filter 的校验与寻址)
- `tests/test_ellipse_filter.py` — EWA 单元测试与独立暴力宽盒参考
- `tests/test_ripmap.py` — ripmap 单元测试与原图面积积分独立参考
- `demo.py` — EWA 固定输入演示:点采样 vs 足迹滤波 + budget/非法输入拒绝
- `demo_iteration.py` — ripmap 演示:层级尺寸表、点采样/ripmap/精确盒积分对比、
  层级选择记录、退化与非法输入路径
