## 1.前置概念
本节的一些概念需要用到的上一节提到的一些术语，不清楚的可以回去看看。

### 1.1 折扣回报
未来的所有的奖励乘以折扣因子就得到了**折扣回报**，可以用下面的式子来表示:

$ U_t = R_t + γ · R_{t+1} + γ^2 · R_{t+2} · · · + γ^{n−t} · R n . $

由于$ \gamma $是一个小于1的常数，因此距离现在越远的奖励$ R $的加权值越小，这种带有折扣因子的回报计算方式得到的就是折扣回报。正常情况下我们都是使用折扣回报而不是单纯的回报。$ \gamma $是我们人为设置的一个参数，其越大，折扣得越慢，说明我们更加看重未来的奖励。其越小，折扣得越快，说明我们更加看重近期的奖励。

### 1.2 最优动作价值函数
动作价值函数表示在当前状态$ s_t $下，直到训练回合结束前，执行动作$ a_t $并且后续一直使用当前的策略$ \pi $来根据状态$ s_{t+1},s_{t+2}... $执行动作$ a_{t+1},a_{t+2}... $<font style="color:#DF2A3F;">（注意，这里的后续的所有状态和动作都是预测，并不是真的执行了动作并转移了状态，它们都是根据策略和状态转移函数算出来的，这个思想类似于MPC）</font>所能带来的回报的平均值<font style="color:#DF2A3F;">（因为回报是一个概率空间，其不确定性来自于策略</font>$ \pi $<font style="color:#DF2A3F;">和状态转移函数，它不是一个确切的数字，我们只能用平均值来表示回报大小，也就是大概能带来多少回报）</font>。动作价值函数的数学表达式如下：

$ Q_π(s_t , a_t) = E[U_t|S_t = s_t , A_t = a_t] $

刚刚使用计算平均值的方法消除了转移函数的不确定性，但是策略$ \pi $仍然是一个变量。我们的目的是要让奖励最大，所以可以定义一个**最优动作价值函数**，来反应如何让奖励最大的同时消除策略这个变量。最优动作价值函数的数学表达式是：

$ Q_{\star}(s_t, a_t) = \max_{\pi} Q_{\pi}(s_t, a_t), \quad \forall s_t \in \mathcal{S}, \; a_t \in \mathcal{A}.
 $

这个函数的意思是在状态$ s_t $下，采取动作$ a_t $，不管当前的策略$ \pi $是什么，后续所能带来的最大的回报的平均值是绝对不会超过$ Q_* $的。其表示的是最大的可能的回报，也就是说，如果能够得到$ Q_* $的话，我们就可以根据哪个动作的$ Q_* $更大去选择动作$ a $。

问题在于如何得到这个$ Q_* $，办法是使用深度Q网络（Deep Q Network，也就是DQN）。



## 2.DQN思想
DQN的基本思想是通过深度学习得到一个神经网络去模拟$ Q_* $，得到的神经网络记作$ Q(s,a;w) $，其中$ w $是神经网络中的参数，也就是我们需要拟合的部分。参数$ w $一开始都是随机给的，通过多次训练，得到一组参数$ w $使得$ Q(s,a;w) $非常接近$ Q_* $<font style="color:#DF2A3F;">(注意，神经网络学习的是最优动作价值函数</font>$ Q_*(s,a) $<font style="color:#DF2A3F;">而不是动作价值函数</font>$ Q(s,a) $<font style="color:#DF2A3F;">)</font>，得到$ w $后根据这个神经网络对不同的动作的打分来判断哪个动作更好，然后就执行这个动作。DQN的神经网络结构大概如图所示：

![](https://cdn.nlark.com/yuque/0/2024/png/49555563/1730094805416-caad948c-720b-43dc-9bd2-b1d83c07a12b.png)DQN的输入是当前的状态$ S $，输出是每一个可能的动作的Q的最大值（也就是它们各自的$ Q_* $）。如果有三个动作（上图所示），就会分别输出三个动作的最大Q值。你会发现这里跟之前的神经网络结构比起来少了一个Softmax函数，因为使用DQN的话策略是直接选择Q值最大的那个动作进行执行（<font style="color:#DF2A3F;">贪婪算法，只选最大的</font>），所以它没有用Softmax函数将其转换成概率空间。

注意，$ Q_*(s_t,a_t) $输出的是在状态$ s_t $下执行动作$ a_t $所能带来的最大的Q值，是一个标量，而DQN的输出是一个向量，包含每一个可能得动作和它对应的$ Q_* $。也就是说，$ Q(s_t,a_t;w) $是一个标量，它是DQN输出里面与动作$ a_t $对应的那个元素。具体关系是：

$ DQN=[ Q(s, a_1; w); Q(s, a_2; w); \dots, Q(s, a_n; w) ] $

其中n是可以执行的动作的个数，这个区别对后续的理解很重要。

## 3.TD算法更新$ w $
知道DQN的结构和作用后，现在的问题是如何去训练这个神经网络，使其与真实的$ Q_* $接近呢？这里先要了解**梯度**的概念。

在之前神经网络章节的时候提到过了梯度的概念，如果把需要被训练的函数$ Q(s,a;w) $设置成目标函数$ L $，然后关于神经网络参数$ w $求梯度的数学表达式是：

$ \nabla_{w} Q(s,a;w) 
\triangleq \frac{\partial Q(s,a;w)}{\partial w} $

由于$ Q(s,a;w) $是一个标量（<font style="color:#DF2A3F;">你已经指定了动作</font>$ a $<font style="color:#DF2A3F;">，则其价值就是一个数字，所以是标量。DQN的输出是一个向量是因为它没有指定动作，所以它输出了所有的动作的价值</font>），因此梯度的形状与$ w $完全相同。

如何通过梯度去更新$ w $来使得其能够使神经网络准确反应模型？就要用到之前的**梯度下降**算法。

假如现在要训练一个模型去预测北京到上海需要用多少时间。模型先告诉我需要14小时（这个数字在最开始可以是随机生成的），然后我实际跑了一次，发现用了16小时，这个时候我就可以用我的实际实验得到的数据去更新我的预测参数$ w $。

![](https://cdn.nlark.com/yuque/0/2024/png/49555563/1730098395023-5dd96329-8344-4eb0-8701-b66e0c119ac3.png)

可以用到类似于LQR和MPC的最优化思想，设置一个损失函数$ L(w)=\frac{1}{2}[Q(s,d;w)-y]^2 $来表示模型输出和实际输出的差异，然后想办法去最小化这个损失函数，就能使模型输出和实际输出接近。之前提到了可以用梯度下降的方法去最小化目标函数，因此我们设这个损失函数为被求梯度的目标函数，先用神经网络$ Q(s,a;w) $预测大概要花多久，记作$ \hat{q} $。然后每跑完一趟，得到一个实际用时数据$ y $后，对损失函数求$ w $的梯度。

![](https://cdn.nlark.com/yuque/0/2024/png/49555563/1730099393815-ae537d7c-3abd-442e-982d-f5b80a3b39f1.png)

最后用梯度下降算法更新$ w $的值，最终$ w $会让这个$ L $越来越小，也就是模型的输出会越来越接近真实的输出。

![](https://cdn.nlark.com/yuque/0/2024/png/49555563/1730099424277-2b338acf-f734-4cbe-b0b2-47316d04c086.png)

但是这样需要训练完一整个回合才能更新一次参数，效率实在是太低，我们可以在中间设置一些节点，比如位于北京到上海之间的一座城市济南，我先记录我到济南的时候已经用了多久时间，然后再看看神经网络预测的济南到上海需要多久，根据这两个数据的差异就可以更新一次$ w $。这就是**TD**（Temporal difference）算法，每运行一段时间就更新一次参数。

![](https://cdn.nlark.com/yuque/0/2024/png/49555563/1730100217142-c4e18680-7d58-433f-a3e0-21c397b99267.png)具体如何更新呢？我用了4.5小时到达济南，然后模型预测还要花11小时到达上海，也就是一共需要$ \hat{y}=4.5+11=15.5 $小时。这里的$ \hat{y} $比$ \hat{q} $更加可靠，因为它包含了一部分真实的信息（4.5小时）而$ \hat{q} $纯粹是估计的（<font style="color:#DF2A3F;">类似于卡尔曼滤波器的思想，部分真实信息+部分估算信息</font>）。把$ \hat{y} $叫做TD Target。

模型预测的总时间是14小时，所以误差就是$ \hat{q}-\hat{y}=14-15.5=-1.5 $小时。这里的误差$ \delta=\hat{q}-\hat{y} $就是TD Error。有了误差，就可以用梯度下降进行参数更新。这样，我们只是走了一段路就能对参数进行一次更新，效率大大提高。

![](https://cdn.nlark.com/yuque/0/2024/png/49555563/1730101705178-3c82114e-d685-429b-87b2-16afa8f17def.png)

## 4.TD与DQN结合
根据之前对回报$ U_t $的公式，可以发现它有这个特性：

![](https://cdn.nlark.com/yuque/0/2024/png/49555563/1730103248290-04634d28-a5a0-4deb-9319-cb9af086e51c.png)

根据这个特性，就可以得到最优**贝尔曼方程**（推导不用管，但是得知道这个特性，对推导感兴趣可以看书）。这个贝尔曼方程很重要，后面经常用得到。

![](https://cdn.nlark.com/yuque/0/2024/png/49555563/1730103403645-1008e46d-b9ee-4aaf-81ef-4076ff0d11d7.png)其中$ maxQ_*(S_{t+1},A;w) = Q_*(S_{t+1},A_{t+1};w) $，意思是在下一个状态$ S_{t+1} $下，采取能够使$ Q_* $最大的动作$ a_{t+1} $所带来的回报（DQN包含了很多个$ Q(s,a;w) $，有多少个动作就有多少个$ Q(s,a;w) $）。

最优贝尔曼方程将当前状态的价值表达为**当前即时奖励**和**未来状态的价值**之和。当智能体在状态$ s_t $下根据已经算出来的Q执行了动作$ a_t $后，根据状态转移函数$ p_t(s' \mid s, a) $进入状态$ s_{t+1} $，然后就可以得到一个我们认为设置的奖励$ R_t $，可以记作$ r_t $，因此我们能够使用的四个变量的四元组就是$ (s_t,a_t,r_t,s_{t+1}) $。

由于这里存在求期望的项，我们写代码做计算是算不出来期望的，因此要想办法把期望项去掉。因此要用到**蒙特卡洛近似**。

蒙特卡洛近似的核心思想是：**通过大量随机样本的均值来估计总体的期望值**。如果想估计一个变量 $ X $ 的期望值 $ E[X] $，我们可以：

1. 从 $ X $ 的分布中采样多个样本 $ x_1, x_2, \dots, x_N $；
2. 计算这些样本的均值，作为期望的近似值：$ \mathbb{E}[X] \approx \frac{1}{N} \sum_{i=1}^N x_i $

简单来说就是有一个概率很复杂的东西，它的平均值很难算出来，如果我想求它的平均值，那我就实际做很多次实验，根据实验结果来计算平均值。

根据贝尔曼公式和蒙特卡洛近似之后可以得到这个式子（不需要管数学推导）：

![](https://cdn.nlark.com/yuque/0/2024/png/49555563/1730103710074-4cebfcec-2a62-4e88-9b05-4a77bd53d29a.png)

也就是说，当前阶段的$ Q_* $可以表示为，当前阶段获得的奖励$ r_t $加上折扣因此乘以下一个阶段的$ Q_* $。到这里就和刚刚提到的从北京到上海的例子很像了，左边的$ Q_* $完全是预测值，它是最早预测的回报。右边的值包含了部分真实值$ r_t $和衰减后的下一步开始的预测值$ Q_*(s_{t+1},a) $,$ r_t $就类似于到达济南后已经观测到的消耗掉的时间，而$ Q_*(s_{t+1},a) $就类似于到达济南后模型预测的剩余的时间。

![](https://cdn.nlark.com/yuque/0/2024/png/49555563/1730104144585-2a84d0f2-2025-44b1-a99c-ce29bac8c121.png)我们使用的是神经网络$ Q(s,a;w) $去拟合$ Q_* $<font style="color:#DF2A3F;">（再次注意，拟合的是最优动作价值函数</font>$ Q_* $<font style="color:#DF2A3F;">而不是动作价值函数</font>$ Q $<font style="color:#DF2A3F;">）</font>因此对于神经网络来说，表达式如下：

![](https://cdn.nlark.com/yuque/0/2024/png/49555563/1730105275130-ab579c4a-0792-4b7a-b354-b229ccd2e2f6.png)

跟之前的开车问题类似，左边的$ Q(s_t,a;w) $完全是预测值，它是一个神经网络的输出。右边的值包含了部分真实值$ r_t $和衰减后的下一步开始的预测值$ Q(s_{t+1},a;w) $，也就是神经网络在t+1时刻预测的$ Q(s_t,a;w) $。右边部分的$ \hat{y} $比左边部分的$ \hat{q} $更加可靠，因此鼓励$ \hat{q} $去靠近$ \hat{y} $，剩下的步骤就和之前提到的梯度下降一样了，用$ \hat{q} $和$ \hat{y} $的差值去更新$ w $。

## 5.流程总结
1. 首先根据贝尔曼公式下面的方式我们可以得到一个四元组$ (s_t,a_t,r_t,s_{t+1}) $。
2. 根据这个四元组和当前的参数$ w_t $我们可以得到DQN的观测Q值：$ \hat{q_t}=Q(s_t,a_t;w_t) $和$ \hat{q_{t+1}}=maxQ(s_{t+1},a;w_t)=Q(s_{t+1},a_{t+1};w_t) $<font style="color:#DF2A3F;">(注意，</font>$ a_{t+1} $<font style="color:#DF2A3F;">是在状态</font>$ s_{t+1} $<font style="color:#DF2A3F;">下根据已经训练出来的神经网络</font>$ Q(s_t,a_t;w_t) $<font style="color:#DF2A3F;">做出的可以使</font>$ Q_* $<font style="color:#DF2A3F;">最大的动作。因为这里我们已经有一个关于所有动作对应的Q值的DQN神经网络，因此它是一个可以被算出来的确定的动作)</font>
3. 计算TD Target：$ \hat{y_t}=r_t+\gamma\hat{q_{t+1}} $。
4. 将他们相减可以得到TD Error：$ \delta_t = \hat{q_t}-\hat{y_t} $。
5. 根据TD Error计算梯度$ g_t = \nabla_w Q(s_t,a;w_t) $
6. 使用梯度下降更新参数$ w_{t+1}=w_t-\alpha\delta g_t $



## 6.额外补充
### 6.1 策略选择
在训练DQN的过程中，我们可以使用任意的策略$ \pi $去控制智能体与环境进行交互，一个常用的办法是$ \epsilon -greedy $**策略**。设置一个变量$ \epsilon $，以概率$ 1-\epsilon $去采取可以使$ Q_* $最大的动作，以概率$ \epsilon $去随机执行动作。

$ \epsilon $越大，探索的概率就越大，机器人越容易找到最优解，但是训练越慢。$ \epsilon $越小，机器人容易陷入局部最优，但是训练会更快。在开始训练的时候，一般设置$ \epsilon $很大，鼓励机器人多尝试不同的道路。随着训练进行，逐渐减小$ \epsilon $的值以加快训练速度。

### 6.2 经验回放
把智能体在一个训练回合中的所有经历的状态，采取的动作和得到的奖励记作一条轨迹，记作：

$ [s_1,a_1,r_1,s_2,a_2,r_2.....s_n,a_n,r_n] $

类似于这个轨迹，在训练的过程中，把多次训练得到的数据存在一个数组中，在更新参数时从这个数组中随机取四元组来进行梯度下降更新，就是**经验回放**。这个数组就是经验回放数组（replay buffer）。

数组会保存b条训练结果，也就是b个四元数组，这个b是我们人为设置的参数。当数据满了以后，删掉前面的数据，填入新的数据（队列）。一般要等数据吧这个经验回放数组填满才会开始更新DQN。

经验回放可以打破各个序列的相关性，确保多次采样的数据都是独立的。而且可以重复利用收集到的经验。

### 6.3 同策略（on policy）和异策略（off policy）
要了解同策略和异策略的差异，先要了解**行为策略（Behavior policy）**和**目标策略（Target policy）**的概念。

行为策略是指智能体在**训练的过程中采取的策略**。在强化学习中，我们先让智能体不断地与环境进行交互，并记录下观测到的状态，动作和奖励等经验。这个过程中它采取的策略就是行为策略，行为策略用于收集经验。

目标策略就是指在智能体完成训练以后，**开始实际运行所采用的策略**，也是我们想要得到的能够使奖励最大的策略，它是训练的结果。

行为策略和目标测量可以一样，也可以不一样。当行为策略和目标策略一样时，就是同策略。当行为策略和目标策略不一样时，就是异策略。比如这一节DQN中，在训练时，我们采用$ \epsilon -greedy $策略，确保机器人在初期可以探索更多的可能性，在后期可以加快训练速度，这个$ \epsilon -greedy $策略就是行为策略。当训练完成后，策略就变成了只选择能够使$ Q_* $最大的动作进行执行，这个就是目标策略，他们两个是不一样的，因此DQN就是一种异策略。

### 6.4 SARSA算法
更新DQN的神经网络用到的算法叫做Q学习，它的目的是更新最优动作价值函数$ Q_* $。这里提到的SARSA算法是一种用于得到动作价值函数$ Q_\pi $的神经网络，也叫价值网络。**这个算法很重要，下一节的Actor-Critic会用到该算法。**

**训练流程：** 设当前价值网络的参数为$ w_{now} $，当前策略为$ \pi_{now} $。每一轮训练用五元组$ (s_t , a_t , r_t , s_{t+1}, \tilde{a}_{t+1}) $对价值网络参数做一次更新。 

1. 观测到当前状态$ s_t $，根据当前策略做抽样得到动作$ a_t $。 

2. 把$ (s_t , a_t) $输入到价值网络$ q(s,a;w) $中，得到其$ Q_\pi $值，记作$ \hat{q}_t $： 

$ \hat{q}_t=q(s_t,a_t;w_{now}) $

3. 智能体执行动作$ a_t $之后，会得到环境反馈的奖励$ r_t $和新的状态$ s_{t+1} $。 

4. 根据当前策略做抽样得到下一个要执行的动作：$ \tilde{a}_{t+1} ∼ \pi_{now}(\cdot|s_{t+1}) $。注意，$ \tilde{a}_{t+1} $只是假想的动作，智能体不会执行这个动作。 

5. 用价值网络计算$ (s_{t+1} , \tilde{a}_{t+1}) $的价值： 

$ \hat{q}_{t+1}=q(s_{t+1},\tilde{a}_{t+1};w_{now}) $

6. 计算 TD 目标和 TD 误差： 

$ \hat{y}_t=r_t+\gamma\hat{q}_{t+1} $     $ \delta_t=\hat{q}_t-\hat{y}_t $

7. 对价值网络$ q $做反向传播，计算$ q $关于$ w $的梯度：$ \nabla_wq(s_t,a_t;w_{now}) $。 

8. 通过梯度下降对价值网络参数$ w $进行更新： 

$ w_{new}=w_{now}-\alpha\cdot\delta_t\cdot\nabla_wq(s_t,a_t;w_{now}) $

9. 用某种算法更新策略函数，下一节的Actor-Critic会细讲。

