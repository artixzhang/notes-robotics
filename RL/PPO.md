Proximal Policy Optimization, 近端策略优化.

# PPO 的概率分布输出

PPO 是随机性策略, 输出的是动作的概率分布. 

例如对于一个 23 自由度的机器人, Actor 神经网络会有 46 个输出节点 (23 个 $\mu$ 均值, 23 个 $\sigma$ 标准差). 对于每个关节输出一个正态分布.

在系统运行这步时, 程序会从正态分布中随机抽样一个实际数值, 该数值是实际发送给机器人的电机扭矩 $A_t$ .

因此, PPO 不需要 $\epsilon$-greedy, 其探索能力被天然的包含在了从正态分布中随机抽样的动作中. 

在训练初期, 网络输出的 $\sigma$ 很大, 抽样出的动作非常离散, 即探索. 随着训练进行, 网络发现某个 $\mu$ 附近总能拿高分, 于是 $\sigma$ 会逐渐变小, 减少探索.

为了防止 $\sigma$ 过早变为 0 导致停止探索, PPO 的 Loss 里会增加熵奖励, 强制要求 $\sigma$ 保持一定的大小.

# PPO 的工作流程

## 数据采集 Rollout

设定一个 Rollout Buffer, 通常大小设定为 $N$ 步, 例如 $N=1000$ .

Agent 开始在环境中运行.

开始进行一次步: Agent 在某一刻处于状态 $S_t$ , Actor 网络输出动作 $A_t$ , 系统给予即时奖励 $R_{t+1}$ , 状态变为 $S_{t+1}$ .
- 将一次步的到的 $(S_t, A_t, R_{t+1},S_{t+1})$ 存入 Buffer. 
- 存储当前步中, Actor 输出动作的对数概率 $\log \pi_{old}(A_t|S_t)$ . 对数概率在网络更新后, 用于计算新旧概率比值.

对于离散的动作空间, 概率比值就是**概率比值**; 对于连续的动作空间, 概率比值是**概率密度比值**.

正态分布的 PDF 为: $f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$

将 $x, \mu, \sigma$ 代入, 即可得到动作 $A_t$ 的概率密度.

继续进行, 直至获取到 $N$ 步数据. 这 $N$ 步可能是没跑完的轨迹, 也可能是好多局的拼接, 但收集满之后就暂停环境.

在数据收集过程中, 会出现数据采集暂停但尚未获得真实结局, 需要依靠 Critic 预测的价值来代替未来收益, 称为强化学习的 **Bootstrapping** 自举.

环境暂停后不会重置, 而是冻结, 等网络参数更新完毕后, Actor-Critic 会继续从上一轮暂停的地方开始.

## 计算优势 GAE

### 单步优势, TD Error, $\delta$

TD Error 也是单步优势, 衡量某步实际动作 $A_t$ 相较平均水平的表现水平.

**平均水平**是通过 Critic 预测得到的. 我们相信 Critic 预测的是大数定律下的平均预期, 超过 Critic 的预测就是比平均水平好.

Actor 与 Critic 是合作竞争进步的.
- Actor 的目的是产生比 Critic 预期更好的结果.
- Critic 的目的是根据 Actor 进步, 使自己预测的更准.

Critic 通过更准确的预测, 能够为 Actor 提供更有意义的指导 GAE, 从而帮助 Actor 拿到更真实的环境即时奖励 $R$ .

TD Error 的方差很小, 但是偏差会很大, 可能是 Critic 笃定的错误偏差. 尤其在训练开始的初期, 由于 Critic 并未获得足够多正确的数据, 因此状态价值函数非常不准确.

单步 TD Error, 优势, $\delta$ 的计算公式为:

$$
\delta_t = R_{t+1}+\gamma V(S_{t+1}) - V(S_t)
$$

$\gamma$ 是折扣因子, 通常设为 0.99, 表示 未来奖励的折损.

### 广义优势估计, GAE, Generalized Advantage Estimator

为了解决 Critic 的瞎猜, 设计了 GAE.
- TD Error 是单步的估计, 方差小, 偏差大.
- 蒙特卡洛 $G_t$ 走完一整条轨迹的方案方差大, 偏差小.

将二者结合是 GAE 的思想, 将每一步的 TD Error 通过一个衰减系数 $\lambda$ 累加, 得到一个加权的 TD Error 和, 既不至于像 $G_t$ 那样一整局的结果方差太大, 又不像单步 TD 那样偏差太大.

GAE 专门作为优势函数供 Actor 更新使用. Critic 依然使用普通的 TD Error 更新.

GAE 的计算公式为:

$$
\begin{align}
A_t^\text{GAE} & = \displaystyle \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l} \\
& = \displaystyle \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + (\gamma\lambda)^3\delta_{t+3} + \cdots
\end{align}
$$

$A_t^\text{GAE}$ 是优势, 不是动作.

GAE 对越未来的 $\delta$ 越不信任 (权重越小).

超参数:
- $\gamma$ 折扣因子, 决定智能体是否在意长远利益, 对于长期或最后才有明显收益的任务, 需要增大 $\gamma$ .
- $\lambda$ 平滑因子, 决定 GAE 对未来的信赖程度.
	- 若 $\lambda=0$ , 公式退化为 $A_t=\delta_t$ , 即单步 TD Error.
	- 若 $\lambda = 1$ , 公式退化为蒙特卡洛 (MC) 估计.
	- 令 $\lambda = 0.95$ , 平衡方差和偏差, 利用多步未来的真实反馈, 并利用指数衰减压制未来巨大的方差.

递推公式为:

$$
A_t^\text{GAE} = \delta_t + \gamma\lambda A_{t+1}^\text{GAE}
$$

实际工程中, 使用逆向遍历 (从后向前). 当下的总优势 = 当下的单步优势 + (衰减系数 × 未来的总优势).

以 Rollout 1000 步为例:
- 计算第 1000 步的单步 $\delta_{1000}$ , GAE 为 $A_{1000}^\text{GAE} = \delta_{1000}$ .
- 计算第 999 步的单步 $\delta_{999}$ , GAE 为 $A_{999}^\text{GAE} = \delta_{999} + \gamma\lambda\cdot A_{1000}^\text{GAE}$
- $\cdots$

计算完成后, Buffer 中包含状态, 动作, 奖励, $\text{GAE}_t$ , $V_{target}$ .

目标价值 $V_{target} = \text{GAE}+V_{old}$ .

$\text{GAE}$ 代表相比平均水平的优势, $V_{old}$ 代表平均水平.

## 模型更新 Learning

由于 PPO 有 Clip 机制, 保证了更新不会极端, 因此允许一批数据反复训练, 通常一批数据会重复训练 4 到 10 个 Epochs.

首先将 Buffer 中的数据打乱 (Shuffle). 将打乱的数据切片成若干 Mini-Batch.

对每个 Mini-Batch:
- 将数据中的 $S$ 输入给 Actor, 计算新的概率 $\pi_{new}$ .
- 将新概率除以 Buffer 中存储的旧概率 $\pi_{old}$ , 得到概率比值 $r_t(\theta)$ .
- 计算 PPO 的 Clip Loss (负目标得分), 反向传播更新 Actor 的参数 $\theta$ .
- 将数据中的 $S$ 输入给 Critic, 计算现在的预测 $V$ , 与 Buffer 中的 $V_{target}$ 计算 MSE Loss, 反向传播更新 Critic 参数 $w$ .

### Actor 参数 $\theta$ 更新

Actor 的参数 $\theta$ 的更新目标是使目标得分更高, 目标得分 $L^\text{CPI}$ 的计算公式为:

$$
L^\text{CPI}(\theta) = \frac{\pi_{new}(A_t|S_t,\theta)}{\pi_{old}(A_t|S_t)}\times \text{GAE}_{t}
$$

用于优化器迭代的 $\text{Loss} = -[r_t(\theta)\times \text{GAE}]$ . 通过 $\dfrac{\partial Loss}{\partial\theta}$ 更新参数.

#### Clip 机制

为了防止网络更新参数过于极端, 导致新旧动作概率比值过大, PPO 算法设计了 Clip 机制, 裁剪掉过大的变化.

对于 Actor 输出动作的实际最终打分遵循以下公式:

$$
最终得分 = \min\Big(r_t(\theta)\cdot\text{GAE}, \text{clip}(r_t(\theta), 1-\epsilon,1+\epsilon)\cdot\text{GAE}\Big)
$$

其中, $\epsilon$ 是裁剪超参数. 例如, 当 $\epsilon = 0.2$ 时, 超过 $1.2$ 或低于 $0.8$ 的动作概率比值会被裁剪到阈值用于后续梯度下降.

当概率比值超出阈值后, Clip 机制会强行使最终得分变为**常数**, 而常数的导数是 0. 因此 Clip 可以切断梯度传递, 使无论参数 $\theta$ 产生多么极端的变化, $\text{Loss}$ 对 $\theta$ 的导数都是 0.

Clip 机制不是限幅, 而是直接丢弃过于激烈的变化. 当某个 $(S,A,R,S)$ 样本将比值推到过高时, Clip 会直接舍弃这个样本. 只有当比值在限度范围内时, 才会更新参数.

### Critic 参数 $w$ 更新

Critic 的参数 $w$ 的更新目标是使 $\delta$ 更小. 构造的 Loss 函数为 MSE (均方误差):

$$
\text{Loss} = (V(S_t)-V_{target})^2
$$

对其求导, 导数为 $2(V(S_t)-V_{target}) = -2\times\delta$ . 即沿着 MSE Loss 梯度下降的方向就是让 $\delta$ 逼近 0 的方向.

### Mini-Batch 对模型的更新

在模型更新过程中, 虽然没有进行新的 Rollout, 却出现了模型计算出的新概率 $\pi_{new}$ , 旧概率 $\pi_{old}$ , 新预测, 旧预测. 

这是由于在 Mini-Batch 训练的过程中, 模型的参数已经发生变化, 在下一个 Mini-Batch 训练的时候, 更新后的模型做出的预测已与 Rollout 时模型预测的结果不同, 因此需要计算新的预测值.

## 重复

对于一条 Trajectory, 进行了多个 Epochs 的训练后, 清空 Rollout Buffer. 

环境继续运行, 并用更新后的 Actor-Critic 去进行新数据的采集. 随后重复训练.

# PPO 的损失函数

在具体的任务中, Actor 和 Critic 可能会有共有的网络部分, 即某些层是共享的, 总的 Loss 也是由两个部分组成的.

PPO 的总 Loss 计算公式为:

$$
L^\text{PPO} = L^\text{CLIP}(\theta) - c_1L^\text{VF}(w) + c_2S[\pi_\theta](s_t)
$$

其中,
- 策略裁剪损失 $L^\text{CLIP}(\theta)$ 代表 Actor Loss. 希望 Clip 的带截断得分最高, 因此该项为正.
- 价值函数损失 $L^\text{VF}(\theta)$ 代表 Critic Loss. 希望均方误差最小, 因此该项为负.
- 熵正则化项 $S[\pi_\theta](s_t)$ 代表熵, 鼓励策略保持探索, 防止过早陷入局部最优. 希望熵增大, 从而鼓励探索, 因此该项为正.

上述公式实际是总体目标得分, 而不是总体损失, 因为 RL 的目标是使得分最高, 而通常不是损失最低, 在训练时整体取负号变为损失.

$$
\text{Total\_Loss} = -L^\text{CLIP}(\theta) + c_1L^\text{VF}(w) - c_2S[\pi_\theta](s_t)
$$

对于正态分布, 熵只与标准差相关, 而正态分布的熵 $S=\ln(\sigma \sqrt{2\pi e})$ . 通过在总体得分中引入 $S$ , 可以间接控制系统保持输出较大的标准差.

$c_1$ 和 $c_2$ 是超参数权重, 负责调节每项的占比重要性.
- $c_1$ 通常是 0.5, 即 Critic 价值的重要性占一半.
- $c_2$ 通常是 0.01 或更小, 即保持一点探索欲望.
