# 纯策略与纯价值的局限

策略梯度 PG 擅长处理连续动作, 但必须轨迹结束才能学习, 效率极低, 方差极大. 类似闭门造车, 只有估计结束, 才能获得最终的回报, 模型并不清楚轨迹过程中的每个动作实际的优劣.

基于价值的 DQN 可以每一步都学习, 效率极高, 但无法处理连续动作, 只会做选择题, 只能从有限的状态空间或动作空间中挑出最好的, 无法处理备选项无限多的情况.

二者结合形成了强化学习领域中应用最广泛的 Actor-Critic 架构. 各种顶级的 RL 算法, 例如 PPO, SAC, DDPG 等底层都是 Actor-Critic.

Actor-Critic 架构设计的核心目的就是解决彼此的局限.

# Actor 与 Critic

**解决 PG 的问题: 用 Critic 代替真实总回报 $G_t$** 

在纯 PG 中, 更新网络使用的是:

$$
\theta \leftarrow \theta + \alpha \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot G_t
$$

其中 $G_t$ 是一条轨迹结束后的真实总得分. 其问题是方差太大, 一连串的好动作会因为最终的失误而被判断为坏动作.

引入一个 Critic, 不需要等一整条轨迹结束计算 $G_t$ , 而是每走一步都由 Critic 直接打分来代替 $G_t$ 指导 Actor 更新.

**解决 DQN 的问题: 用 Actor 代替 $\max_{a'}$ 操作**

DQN 无法输出连续动作空间的结果.

引入一个策略网络 Actor, 天然可以输出连续动作的概率分布. Critic 只需要对 Actor 选定的动作进行打分, 而不需要选择输出, 彻底规避了遍历寻找 $\max_{a'}$ 的操作.

在 Actor-Critic 结构中, 需要同时训练两个网络.

Actor 策略网络 $\pi_{\theta}$
- 输入: 当前状态 $S_t$ .
- 输出: 选取动作 $A_t$ . 
	- 若为离散动作 (超级玛丽): 输出不同按键的概率分布, 例如 $[上: 0.1, 下: 0.1, 左: 0.7, 右: 0.1]$ .
	- 若为连续动作 (23 自由度机器人): 输出 23 个正态分布的均值 $\mu$ 和标准差 $\sigma$ . 称之为随机性策略.

Critic 网络 $V_{w}$ 或 $Q_{w}$ (主流算法 Critic 通常评估状态价值 $V(S_t)$ )
- 输入: 当前状态 $S_t$
- 输出: 实数, 状态价值.

其中, 下标 $w$ 是 Critic 价值网的网络参数.

# 优势函数

Critic 给 Actor 打分需要设计一些机制, 让打分更合理. 例如, 若直接用 $Q(S,A)$ 给动作打分, 会出现原本状态很好, 动作一般, 却因为状态足够好, 而获得了高分, 这样会让 Actor 误以为是自己的动作选择优秀, 实际只是因为状态好.

因此 Critic 给 Actor 打分的趋势应该是 **Actor 的动作与这个状态下的平均水平相比如何?**

由此定义优势函数:

$$
A(S,A) = \underbrace{Q(S,A)}_{动作产生价值} - \underbrace{V(S)}_{状态平均价值}
$$

- 若 $A>0$ : 说明动作本身比平均水平高, Actor 应当增加该动作出现的概率.
- 若 $A<0$ : 说明动作本身比平均水平低, Actor 应当降低该动作出现的概率.

进一步, 由于 $Q(S,A) = R+\gamma V(S')$ , 因此可以将优势函数改写为:

$$
A(S,A) \approx R+\gamma V(S') - V(S)
$$

其公式与 TD Error 相同. 用 $\delta$ 表示 TD Error:

$$
\delta = R_{t+1} + \gamma V_w(S_{t+1}) - V_w(S_t)
$$

$\delta$ 既作为 Critic 的 $\text{Loss}$ , 又作为优势函数指导 Actor 的评价指标. 

# Actor-Critic 具体工作流程

Actor 网络的参数为 $\theta$ , Critic 网络的参数是 $w$ .

1. Actor 产生动作并执行 (采样):
	- Actor 观察当前状态 $S_t$ .
	- Actor 根据自己的策略网络 $\pi_{\theta}(a|S_t)$ 选出一个动作 $A_t$ 并执行.
2. 环境反馈:
	- 环境给予即时奖励 $R_{t+1}$ , 并更新智能体的状态到 $S_{t+1}$ .
3. Critic 打分与自我更新:
	- Critic 查看数据组 $(S_t,A_t,R_{t+1},S_{t+1})$ .
	- Critic 计算当前 TD Error: $\delta = R_{t+1} + \gamma V_w(S_{t+1}) - V_w(S_t)$ .
	- Critic 自我更新: 预测有误差 $\delta$ 使用梯度下降修改参数 $w$ , 使 $V_w(S_t)$ 逼近 $R_{t+1}+\gamma V_w(S_{t+1})$ .
4. Actor 更新:
	- Actor 根据 Critic 给出的 $\delta$ (优势函数) 进行判断.
	- 若 $\delta>0$ , 则表现良好, Actor 朝增大 $A_t$ 出现概率的方向更新.
	- 若 $\delta<0$ , 则表现糟糕, Actor 朝减小 $A_t$ 出现概率的方向更新.
	- Actor 的更新公式为: $\theta \leftarrow \theta + \alpha \nabla_{\theta}\log\pi_{\theta}(A_t|S_t)\times \delta$ .
5. 循环:
	- 进入下一步 $S_{t+1}$ , 重复上述过程, 直至结束.

使用 Critic 用 TD 算法每一步都能计算出 $\delta$ (TD Error), 因此 Actor 不再需要等待一局结束再计算 $G_t$ , 而是每走一步, 环境给出一个 $R_{t+1}$ , Critic 立刻就能计算 $\delta$ .

在现代主流的 AC 算法中, 基本都会引入目标网络, 来使训练更加稳定, 并且在 Critic 和 Actor 中均有出现.

以 DDPG 算法为例, 系统中共维护了 4 个神经网络:
- Actor 当前网络 ( $\theta$ ): 负责实际选取动作.
- Actor 目标网络 ( $\theta^-$ ): 负责在算目标值时提供未来动作.
- Critic 当前网络 ( $w$ ): 负责计算 $\delta$ , 并更新自己.
- Critic 目标网络 ( $w^-$ ): 负责计算目标值 $R_{t+1} + \gamma V_{w^-}(S_{t+1})$ , 冻结网络.

DDPG 中的目标网络并不像 DQN 那样周期性的硬拷贝, 而是采用软更新的方式, 每一步都将当前网络的参数向目标网络中融合一点:

$$
w^- \leftarrow \tau w + (1-\tau)w^-
$$

其中, $\tau$ 是一个很小的值.

