TRPO(Trust region policy optimization)算法新引入了一个概念-**置信域**。我们机器人用到的PPO算法和TRPO算法的提出者是同一个人，两者十分相似，因此理解TRPO对理解PPO很有帮助。

## 1.置信域的定义
强化学习的策略训练过程本质上是找到一个参数![image](https://cdn.nlark.com/yuque/__latex/ed5a4aa5e092e303a69c608582c70db9.svg)使得目标函数![image](https://cdn.nlark.com/yuque/__latex/ca2bfb8e73d65f76f8e20c82a070c0e2.svg)最大，之前用的是梯度上升算法使其最大，而置信域就是另一种办法使其最大。

置信域的定义是：给定当前的参数![image](https://cdn.nlark.com/yuque/__latex/ef66567f60da4269a1e21a5dade88d3a.svg)，在![image](https://cdn.nlark.com/yuque/__latex/ef66567f60da4269a1e21a5dade88d3a.svg)附近的领域![image](https://cdn.nlark.com/yuque/__latex/8bc60897355507a044182e4548171a69.svg)中，构造一个函数![image](https://cdn.nlark.com/yuque/__latex/e12eedf392d72140ea2eb12777b61640.svg)，这个函数要满足这个条件：

![image](https://cdn.nlark.com/yuque/__latex/e64110303caa1a2503658734e6d2daa7.svg)  ![image](https://cdn.nlark.com/yuque/__latex/5e6a4616bd5ed253d29e884055a11e88.svg)

其中关于![image](https://cdn.nlark.com/yuque/__latex/8bc60897355507a044182e4548171a69.svg)就是置信域，也就是说在![image](https://cdn.nlark.com/yuque/__latex/8bc60897355507a044182e4548171a69.svg)中（![image](https://cdn.nlark.com/yuque/__latex/ef66567f60da4269a1e21a5dade88d3a.svg)的附近），我们可以信任![image](https://cdn.nlark.com/yuque/__latex/e12eedf392d72140ea2eb12777b61640.svg)并用它来代替![image](https://cdn.nlark.com/yuque/__latex/308ed96aa201ec8fe47df94d5afafe88.svg)。![image](https://cdn.nlark.com/yuque/__latex/8bc60897355507a044182e4548171a69.svg)的一个形象一点的例子就是以![image](https://cdn.nlark.com/yuque/__latex/ef66567f60da4269a1e21a5dade88d3a.svg)为圆心，以一个半径画圆：

![](https://cdn.nlark.com/yuque/0/2024/png/49555563/1730355197966-a9ce1634-4ae3-4cc7-924c-d18e7c23609e.png)

用一个一元函数来举例![image](https://cdn.nlark.com/yuque/__latex/e12eedf392d72140ea2eb12777b61640.svg)和![image](https://cdn.nlark.com/yuque/__latex/308ed96aa201ec8fe47df94d5afafe88.svg)的关系，如下图所示。其中绿色的线是指原来的目标函数![image](https://cdn.nlark.com/yuque/__latex/308ed96aa201ec8fe47df94d5afafe88.svg)，紫色的线是我们人为找到的函数![image](https://cdn.nlark.com/yuque/__latex/c895173d3be4872abf206be4268a58cb.svg)。可以看到在![image](https://cdn.nlark.com/yuque/__latex/ef66567f60da4269a1e21a5dade88d3a.svg)附近的置信域里，![image](https://cdn.nlark.com/yuque/__latex/e64110303caa1a2503658734e6d2daa7.svg)。但是一旦超出置信域，![image](https://cdn.nlark.com/yuque/__latex/56a6d564c18c374fa54fa5976da28226.svg)和![image](https://cdn.nlark.com/yuque/__latex/308ed96aa201ec8fe47df94d5afafe88.svg)的差异就可能会很大。<font style="color:#DF2A3F;">这里有点像传统控制里的线性化，只有在平衡点附近才能对其进行线性化，超出一定范围就会失真。</font>

![](https://cdn.nlark.com/yuque/0/2024/png/49555563/1730355467583-60bbcb2d-8662-460a-b5fc-99f7f7bd7242.png)

通常来说![image](https://cdn.nlark.com/yuque/__latex/308ed96aa201ec8fe47df94d5afafe88.svg)是一个很复杂的函数（我们之前为了对![image](https://cdn.nlark.com/yuque/__latex/308ed96aa201ec8fe47df94d5afafe88.svg)求偏导做了那么多数学近似），但是![image](https://cdn.nlark.com/yuque/__latex/c895173d3be4872abf206be4268a58cb.svg)可以很简单，它可以是![image](https://cdn.nlark.com/yuque/__latex/308ed96aa201ec8fe47df94d5afafe88.svg)的近似，也可以是![image](https://cdn.nlark.com/yuque/__latex/308ed96aa201ec8fe47df94d5afafe88.svg)在![image](https://cdn.nlark.com/yuque/__latex/ef66567f60da4269a1e21a5dade88d3a.svg)附近的泰勒展开。<font style="color:#DF2A3F;">其实就是类似于以</font>![image](https://cdn.nlark.com/yuque/__latex/ef66567f60da4269a1e21a5dade88d3a.svg)<font style="color:#DF2A3F;">为平衡点对</font>![image](https://cdn.nlark.com/yuque/__latex/308ed96aa201ec8fe47df94d5afafe88.svg)<font style="color:#DF2A3F;">进行线性化处理。</font>

在找到![image](https://cdn.nlark.com/yuque/__latex/c895173d3be4872abf206be4268a58cb.svg)后，就在置信域里找到新的参数![image](https://cdn.nlark.com/yuque/__latex/c5d9eacbcefd7d8091dbfcac92ad5a3d.svg)对![image](https://cdn.nlark.com/yuque/__latex/c895173d3be4872abf206be4268a58cb.svg)进行最大化（因为我们本质上是想对![image](https://cdn.nlark.com/yuque/__latex/308ed96aa201ec8fe47df94d5afafe88.svg)进行最大化，在置信域里，![image](https://cdn.nlark.com/yuque/__latex/e64110303caa1a2503658734e6d2daa7.svg)，所以对![image](https://cdn.nlark.com/yuque/__latex/c895173d3be4872abf206be4268a58cb.svg)进行最大化就是对![image](https://cdn.nlark.com/yuque/__latex/308ed96aa201ec8fe47df94d5afafe88.svg)进行最大化）：

![](https://cdn.nlark.com/yuque/0/2024/png/49555563/1730356031085-b468af0e-d38f-4f30-a634-0cff1f67096d.png)

## 2.TRPO算法
TRPO就是利用置信域，根据之前的策略的表现来更新现在的策略。TRPO算法的流程第一步是做近似，第二部是最大化。

### 2.1 做近似
我们可以对状态价值函数分子分母同乘以一个![image](https://cdn.nlark.com/yuque/__latex/43848218d418936baaea7901aaa313d9.svg)，做一个变形：

![](https://cdn.nlark.com/yuque/0/2024/png/49555563/1730356630315-ca3ee11d-fb72-44df-8663-40beacf09912.png)

这样目标函数就可以写成：

![](https://cdn.nlark.com/yuque/0/2024/png/49555563/1730356759750-52164928-3d14-4cf9-9ffe-e0c119d11ef8.png)

根据这个定理，我们就可以找出![image](https://cdn.nlark.com/yuque/__latex/c895173d3be4872abf206be4268a58cb.svg)。

要做计算最忌讳的就是求期望，凡是存在期望的公式我们都不能直接使用，需要想办法**把期望消掉**。之前我们用蒙特卡洛近似来消掉期望，这里也可以用。

首先用当前的策略网络![image](https://cdn.nlark.com/yuque/__latex/78ad1b452ecaf138bca9476cb02d5a3f.svg)控制智能体与环境进行交互，从头到尾完成一次训练，就可以得到一条轨迹：![image](https://cdn.nlark.com/yuque/__latex/894fd2df417d21874c57338758ae3a0e.svg)。使用蒙特卡洛近似，可以用实际实验数据来近似期望，因此对这些从多次实际运行中得到的数据求均值就可以大致估计期望值，这样我们就可以设置函数![image](https://cdn.nlark.com/yuque/__latex/c895173d3be4872abf206be4268a58cb.svg):

![](https://cdn.nlark.com/yuque/0/2024/png/49555563/1730357190445-71924330-8fe9-497e-ad72-97354dd92f0d.png)

现在这个![image](https://cdn.nlark.com/yuque/__latex/c895173d3be4872abf206be4268a58cb.svg)还是不能直接用，因为存在![image](https://cdn.nlark.com/yuque/__latex/0a522f8ece17a1ef4ec7d1cdbdca9045.svg)项，还需要对![image](https://cdn.nlark.com/yuque/__latex/0a522f8ece17a1ef4ec7d1cdbdca9045.svg)项进行近似处理。

类似于之前REINFORCE中的思路，可以用![image](https://cdn.nlark.com/yuque/__latex/51b3390153fd75978bd368d2e9a502cf.svg)来对![image](https://cdn.nlark.com/yuque/__latex/0a522f8ece17a1ef4ec7d1cdbdca9045.svg)项做蒙特卡洛近似。

![](https://cdn.nlark.com/yuque/0/2024/png/49555563/1730358172895-6541a786-df61-4ab6-9255-a22492678c37.png)

但是，轨迹中得到的奖励信息![image](https://cdn.nlark.com/yuque/__latex/d7e6f617e901e8dee93551898a5744b7.svg)都是使用旧的策略（也就是上个周期得到的参数所控制的策略）![image](https://cdn.nlark.com/yuque/__latex/4c91a10a309a9fdce94f94c3392b9c30.svg)得到的，而公式中的![image](https://cdn.nlark.com/yuque/__latex/0a522f8ece17a1ef4ec7d1cdbdca9045.svg)需要使用策略![image](https://cdn.nlark.com/yuque/__latex/ab50e673670f5c97b23a8d0e3a382fff.svg)来计算（也就是这个周期的参数所控制的策略，但是这个周期的参数是不确定的，它是个变量），两者是不一致的。因此，只有在这个周期的参数![image](https://cdn.nlark.com/yuque/__latex/ed5a4aa5e092e303a69c608582c70db9.svg)和上一个周期的参数![image](https://cdn.nlark.com/yuque/__latex/ef66567f60da4269a1e21a5dade88d3a.svg)非常接近的情况下，才能用![image](https://cdn.nlark.com/yuque/__latex/51b3390153fd75978bd368d2e9a502cf.svg)来对![image](https://cdn.nlark.com/yuque/__latex/0a522f8ece17a1ef4ec7d1cdbdca9045.svg)项做蒙特卡洛近似。**这就是置信域起作用的地方**，只用当![image](https://cdn.nlark.com/yuque/__latex/ed5a4aa5e092e303a69c608582c70db9.svg)位于参数![image](https://cdn.nlark.com/yuque/__latex/ef66567f60da4269a1e21a5dade88d3a.svg)的置信域内，才能做近似。经过两次近似后，![image](https://cdn.nlark.com/yuque/__latex/308ed96aa201ec8fe47df94d5afafe88.svg)函数就变成了：

![](https://cdn.nlark.com/yuque/0/2024/png/49555563/1730358320973-d46595ef-ed58-4cce-8b97-3c82d024b6d6.png)

其中![image](https://cdn.nlark.com/yuque/__latex/6d71b640d84fc8d05afdb93eaef573f9.svg)叫做**重要性采样比率**![image](https://cdn.nlark.com/yuque/__latex/4ba2313b02bf2984b153fa0441a58a38.svg)，它的物理意义是：新的策略![image](https://cdn.nlark.com/yuque/__latex/d55c28d85ffe2516652e7b77c7dcd78c.svg)在状态![image](https://cdn.nlark.com/yuque/__latex/e24a254996c6d6d65ed16befdaac934d.svg)下采取动作![image](https://cdn.nlark.com/yuque/__latex/c61a8b387e1cb6c40608f4ae65d6f6a6.svg)的概率与旧的策略![image](https://cdn.nlark.com/yuque/__latex/f6f931d8c0afd1c2fe5c27da6256f9f0.svg)在状态![image](https://cdn.nlark.com/yuque/__latex/e24a254996c6d6d65ed16befdaac934d.svg)下采取动作![image](https://cdn.nlark.com/yuque/__latex/c61a8b387e1cb6c40608f4ae65d6f6a6.svg)的概率的比值。重要性采样比率大于1，则说明新的策略会增加在状态![image](https://cdn.nlark.com/yuque/__latex/e24a254996c6d6d65ed16befdaac934d.svg)下采取动作![image](https://cdn.nlark.com/yuque/__latex/c61a8b387e1cb6c40608f4ae65d6f6a6.svg)的概率，反之亦然。

还有一种做法是用上一节提到的优势函数![image](https://cdn.nlark.com/yuque/__latex/5a6f6c8d04e460853d32c4a58b737185.svg)来代替![image](https://cdn.nlark.com/yuque/__latex/51b3390153fd75978bd368d2e9a502cf.svg)。一般是使用广义优势估计（GAE）来计算优势：

![image](https://cdn.nlark.com/yuque/__latex/1c9868bde0c649ec26faac4621a16b4f.svg)

其中![image](https://cdn.nlark.com/yuque/__latex/29dac1bb46607a9859d5438c8b831c80.svg)是TD Error，![image](https://cdn.nlark.com/yuque/__latex/4aa418d6f0b6fbada90489b4374752e5.svg)是衰减因子，![image](https://cdn.nlark.com/yuque/__latex/e520c061a407db472027709bf3f73290.svg)是优势衰减因子。

总结一下，用![image](https://cdn.nlark.com/yuque/__latex/c895173d3be4872abf206be4268a58cb.svg)去近似![image](https://cdn.nlark.com/yuque/__latex/308ed96aa201ec8fe47df94d5afafe88.svg)，然后在置信域内用![image](https://cdn.nlark.com/yuque/__latex/05b66f7a173e71382df071759653e791.svg)去近似![image](https://cdn.nlark.com/yuque/__latex/c895173d3be4872abf206be4268a58cb.svg)，所以就是在置信域内用![image](https://cdn.nlark.com/yuque/__latex/05b66f7a173e71382df071759653e791.svg)去近似![image](https://cdn.nlark.com/yuque/__latex/308ed96aa201ec8fe47df94d5afafe88.svg)。

### 2.2 最大化
最大化就是在![image](https://cdn.nlark.com/yuque/__latex/ef66567f60da4269a1e21a5dade88d3a.svg)的置信域内找到一个![image](https://cdn.nlark.com/yuque/__latex/ed5a4aa5e092e303a69c608582c70db9.svg)，使得![image](https://cdn.nlark.com/yuque/__latex/e8029faa18ddc47306a82c4f23707822.svg)最大。

置信域的取法很多，一种是刚刚提到的那个形象的例子，直接设置一个半径，以![image](https://cdn.nlark.com/yuque/__latex/ef66567f60da4269a1e21a5dade88d3a.svg)为圆心，位于半径内的![image](https://cdn.nlark.com/yuque/__latex/ed5a4aa5e092e303a69c608582c70db9.svg)都是处于它的置信域。这种情况下的最大化公式就是：

![](https://cdn.nlark.com/yuque/0/2024/png/49555563/1730358936719-daad3c76-0021-45a4-8f9a-aa04d0f7c12c.png)

还有一种就是用KL散度来描述两个函数的差异。不需要知道KL散度的原理，只需要知道KL散度用来描述两个概率密度函数的差异。![image](https://cdn.nlark.com/yuque/__latex/870edf8e62803ed11b09a1028e91a279.svg)和![image](https://cdn.nlark.com/yuque/__latex/9fba416e5c94a3f8b4b093c1e6a2ead7.svg)的差异越大，它们之间的KL散度越大。在这种情况下，最大化公式就变成了：

![](https://cdn.nlark.com/yuque/0/2024/png/49555563/1730359107991-d891d513-fdee-4bb9-b310-c768914c68a4.png)

KL散度的效果比用球体好，一般都是用KL散度。



### 2.3 TRPO理解
**1.逻辑层面**

从逻辑上去理解的话，其实就是根据对其他策略（之前的策略）的表现来改进自己的策略（现在的策略）。比如老师上课表扬认真听讲的同学，那自己也可以尝试去多认真听讲来获得奖励。老师批评上课开小差的同学，自己也应该少开小差来获取更多的奖励。但是这个根据的对象应该和自己接近，不然就没有参考价值。比如老师表扬了一名坏学生，说他今天上课没有动手打老师，但是自己从来都不可能会动手打老师，所以这个不打老师的行为对自己没有任何学习价值，因此要设置一个限定范围，只学习跟自己差不多的学生。

**2.数学层面**

从数学上去理解的话，就是要最大化重要性采用比率与优势函数的乘积![image](https://cdn.nlark.com/yuque/__latex/6e1c5f5e48264c9d04fd7a6c5f0d197e.svg)。刚刚提到，重要性采用比率![image](https://cdn.nlark.com/yuque/__latex/4ba2313b02bf2984b153fa0441a58a38.svg)的物理意义是：新的策略![image](https://cdn.nlark.com/yuque/__latex/d55c28d85ffe2516652e7b77c7dcd78c.svg)在状态![image](https://cdn.nlark.com/yuque/__latex/e24a254996c6d6d65ed16befdaac934d.svg)下采取动作![image](https://cdn.nlark.com/yuque/__latex/c61a8b387e1cb6c40608f4ae65d6f6a6.svg)的概率与旧的策略![image](https://cdn.nlark.com/yuque/__latex/f6f931d8c0afd1c2fe5c27da6256f9f0.svg)在状态![image](https://cdn.nlark.com/yuque/__latex/e24a254996c6d6d65ed16befdaac934d.svg)下采取动作![image](https://cdn.nlark.com/yuque/__latex/c61a8b387e1cb6c40608f4ae65d6f6a6.svg)的概率的**比值**。若重要性采样比率大于1，则说明新的策略会增加在状态![image](https://cdn.nlark.com/yuque/__latex/e24a254996c6d6d65ed16befdaac934d.svg)下采取动作![image](https://cdn.nlark.com/yuque/__latex/c61a8b387e1cb6c40608f4ae65d6f6a6.svg)的概率。优势函数![image](https://cdn.nlark.com/yuque/__latex/598e6b29b59affe4329394b9e23854d0.svg)表示在在状态![image](https://cdn.nlark.com/yuque/__latex/e24a254996c6d6d65ed16befdaac934d.svg)下采取动作![image](https://cdn.nlark.com/yuque/__latex/c61a8b387e1cb6c40608f4ae65d6f6a6.svg)带来的相对优势，当![image](https://cdn.nlark.com/yuque/__latex/55be2425d58fb81a1a2dd17620704148.svg)时，采取动作![image](https://cdn.nlark.com/yuque/__latex/c61a8b387e1cb6c40608f4ae65d6f6a6.svg)会大于采取其他动作带来的优势的平均水平。

因此，优化目标通过![image](https://cdn.nlark.com/yuque/__latex/6e1c5f5e48264c9d04fd7a6c5f0d197e.svg)来引导新策略：

+ 增加优势值![image](https://cdn.nlark.com/yuque/__latex/ef641895ed9c2fa09cb0219e95566039.svg)的动作概率。
+ 减少优势值![image](https://cdn.nlark.com/yuque/__latex/a0d04803c88c9643624fb51cfd24e2b3.svg)的动作概率。

当我们最大化![image](https://cdn.nlark.com/yuque/__latex/6e1c5f5e48264c9d04fd7a6c5f0d197e.svg)时，实质上是在**调整新策略的选择倾向**，以增加对好的动作的偏好，减少对差的动作的选择。以下是具体原因：

+ **如果**![image](https://cdn.nlark.com/yuque/__latex/ef641895ed9c2fa09cb0219e95566039.svg)：意味着在状态![image](https://cdn.nlark.com/yuque/__latex/e24a254996c6d6d65ed16befdaac934d.svg)执行动作![image](https://cdn.nlark.com/yuque/__latex/c61a8b387e1cb6c40608f4ae65d6f6a6.svg)的效果优于平均水平。最大化![image](https://cdn.nlark.com/yuque/__latex/6e1c5f5e48264c9d04fd7a6c5f0d197e.svg)会**增加比率 **![image](https://cdn.nlark.com/yuque/__latex/4ba2313b02bf2984b153fa0441a58a38.svg)，即新策略的概率![image](https://cdn.nlark.com/yuque/__latex/d55c28d85ffe2516652e7b77c7dcd78c.svg)相对旧策略的概率![image](https://cdn.nlark.com/yuque/__latex/d55c28d85ffe2516652e7b77c7dcd78c.svg)提升。这会使新策略在状态 ![image](https://cdn.nlark.com/yuque/__latex/e24a254996c6d6d65ed16befdaac934d.svg)更倾向于选择动作![image](https://cdn.nlark.com/yuque/__latex/c61a8b387e1cb6c40608f4ae65d6f6a6.svg)，因为它带来了更高的回报。
+ **如果**![image](https://cdn.nlark.com/yuque/__latex/a0d04803c88c9643624fb51cfd24e2b3.svg)：意味着动作![image](https://cdn.nlark.com/yuque/__latex/c61a8b387e1cb6c40608f4ae65d6f6a6.svg)的效果低于平均水平。最大化![image](https://cdn.nlark.com/yuque/__latex/6e1c5f5e48264c9d04fd7a6c5f0d197e.svg)会**减小比率 **![image](https://cdn.nlark.com/yuque/__latex/4ba2313b02bf2984b153fa0441a58a38.svg)，即新策略的概率![image](https://cdn.nlark.com/yuque/__latex/d55c28d85ffe2516652e7b77c7dcd78c.svg)相对于旧策略![image](https://cdn.nlark.com/yuque/__latex/d55c28d85ffe2516652e7b77c7dcd78c.svg)的概率降低。这意味着新策略在状态![image](https://cdn.nlark.com/yuque/__latex/e24a254996c6d6d65ed16befdaac934d.svg)下不再倾向于选择带来较差结果的动作![image](https://cdn.nlark.com/yuque/__latex/c61a8b387e1cb6c40608f4ae65d6f6a6.svg)，从而避免了负面回报。

通过这种方式，新策略逐步调整，增加选择优势值高的动作的概率，减少选择优势值低的动作的概率，从而提升了策略的总体回报。

## 3.PPO算法
PPO算法就是我们机器人用到的算法，也是这个教程最终想要达到的地方。PPO和TRPO由同一个人提出，它们的算法的基本思想，近似方式和更新方式，都差不多，最大的区别就是PPO引入了“截断”的思想。

### 3.1 更新方法的差异：
**TRPO：**

采用了严格的信任域限制，通过优化一个受限的目标函数来控制策略变化幅度。TRPO 的更新公式如下：

![image](https://cdn.nlark.com/yuque/__latex/592347931238dd71547b33fbcfa4493a.svg)

使得策略更新在 KL 散度约束下满足：

![image](https://cdn.nlark.com/yuque/__latex/628a5028ec309be6d717ff165f1eb255.svg)

TRPO 通过求解这个约束优化问题来实现策略的更新。



**PPO：**

PPO 放弃了严格的 KL 散度约束，而是使用一种剪切（Clipping）损失函数来控制策略更新幅度。PPO 的目标是最大化以下损失函数：

![image](https://cdn.nlark.com/yuque/__latex/cd41d8751ad9a841b3fd461cf9e3c178.svg)

其中，![image](https://cdn.nlark.com/yuque/__latex/ae2727237c204989cc137e915b1aaa17.svg) 表示重要性采样比率（新旧策略的比值），![image](https://cdn.nlark.com/yuque/__latex/7c102e7a7d231bf935f9bc23417779a8.svg) 是剪切范围，通常是一个很小的参数，比如0.1，用于控制策略更新幅度。



**Clip 的作用**

剪切操作的作用在于对 ![image](https://cdn.nlark.com/yuque/__latex/4ba2313b02bf2984b153fa0441a58a38.svg) 限制在区间 ![image](https://cdn.nlark.com/yuque/__latex/c778046c55e4b7f3856d85dfb78922dd.svg) 内：

+ 如果 ![image](https://cdn.nlark.com/yuque/__latex/4ba2313b02bf2984b153fa0441a58a38.svg)在 ![image](https://cdn.nlark.com/yuque/__latex/c778046c55e4b7f3856d85dfb78922dd.svg) 区间内，则使用未剪切的项![image](https://cdn.nlark.com/yuque/__latex/6e1c5f5e48264c9d04fd7a6c5f0d197e.svg)，也就是正常更新。
+ 如果 ![image](https://cdn.nlark.com/yuque/__latex/4ba2313b02bf2984b153fa0441a58a38.svg)超出这个区间，则将其剪切为边界值 ![image](https://cdn.nlark.com/yuque/__latex/843612417381ff611b2878a49054d49a.svg)或![image](https://cdn.nlark.com/yuque/__latex/510da74308ebaeae9efa8f59e229f676.svg)，避免策略变化过大。

**直观理解**：剪切操作限制了策略变化的幅度。如果新策略相对旧策略变化过大（比率偏离 1 太多），损失函数会“平坦化”，阻止优化过程进一步增大比率。



### 3.2 PPO的优势
TRPO要求新的策略和旧策略的KL散度严格小于一个限制，在计算这个KL散度的过程中就要消耗大量的计算力，而且这个限制是硬限制，必须严格满足，所以它必须要每次都去计算，直到找到符合要求的新策略。

PPO不需要对新的策略进行严格的KL限制，只要他们的比率大于某个数直接截断就行了，因此计算量就会小很多，收敛速度也会更快。

实际应用中大多都是用PPO而非TRPO。



