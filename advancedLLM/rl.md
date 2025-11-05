# MTP


# Model Weights Update in slime

背景：training engine的model更新好之后要传给sgl，我们考虑colocate

baseline：通过`update_weight_from_tensors`，我们不要做真正数据的拷贝，我们利用`cudaIPChandler`（类似指针），传输很小的指针即可

more#1:从training model获取tensor的时候，我们可以做通信异步（TP中的allgather，PP/EP中的broadcast），可以利用`torch.dist`里的`async_op`，这里我们异步的是不同rank的通信过程

more#2:对sgl的update model weight请求太多，不好，因为http请求太多也会拖慢进度。所以做bucket，组多个tensor到一个buffer，达到bucket size一起传，类似batch传输（这是在优化服务器调用）

more#3:还是太多http api call，可以flatten tensor，把多个tensor组成大tensor，只会带来一些rebuild的overhead。（之前很多overhead来自handler的打开/关闭）

more#4:参数dict太大，我们可以cache下来，包括expert dict等，这是在load weight的优化

# `top_k` `top_p`
这两个参数是控制sample范围的，我们只取logits中prob最高的`top_k`个token，在他们之间做norm之后sample，这是控制greedy程度的参数。

`top_p`控制nucleus sampling，我们只取prob达到`top_p`的token集合，这是控制多样性的。

# REINFORCE

$$
L(\theta) = -A_t \text{log}\pi _{\theta}(a_t)
$$


对softmax的求导:

$$
\nabla \pi _{\theta} (i) = \pi _{\theta} (i) (\delta _{ik} - \pi _{\theta} (k))
$$

设输入向量  $z \in \mathbb{R}^V $，其 softmax 定义为：

$$
\pi(i) = \mathrm{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j} e^{z_j}}.
$$



$$
\frac{\partial \pi(i)}{\partial z_k}
= \pi(i)\big(\mathbf{1}[i=k]-\pi(k)\big).
$$


$$
\nabla_{z}\pi = \mathrm{diag}(\pi) - \pi\,\pi^\top.
$$

---

**推导过程**

$$
\frac{\partial \pi(i)}{\partial z_k}
= \frac{e^{z_i}\cdot \mathbf{1}[i=k]\sum_j e^{z_j} - e^{z_i}\cdot e^{z_k}}
{(\sum_j e^{z_j})^2}
= \frac{e^{z_i}}{\sum_j e^{z_j}}
\Big(\mathbf{1}[i=k]-\frac{e^{z_k}}{\sum_j e^{z_j}}\Big)
= \pi(i)\big(\mathbf{1}[i=k]-\pi(k)\big).
$$

---

**REINFORCE 单步梯度**

单步损失定义为：

$$
\ell_t = -A_t \log \pi(a_t)
$$

对 logits 求导：

$$
\frac{\partial \ell_t}{\partial z_k}
= -A_t \frac{\partial \log \pi(a_t)}{\partial z_k}
= -A_t(\mathbf{1}[a_t=k]-\pi(k))
= A_t(\pi(k)-\mathbf{1}[k=a_t]).
$$

向量形式：

$$
\nabla_{z}\ell_t = A_t(\pi - e_{a_t}),
$$

并且满足归一性质：

$$
\sum_k \frac{\partial \ell_t}{\partial z_k} = A_t(\sum_k \pi(k) - 1) = 0.
$$



# DAPO：Decoupled Clip and Dynamic sAmpling Policy Optimization

| 技术 | 描述 | 目的 |
|------|------|------|
| **Clip-Higher** | 对策略比率 $r_t = \frac{\pi(a_t)}{\pi_{old}(a_t)}$ 使用更高的裁剪上限。 | 防止策略更新过度保守，缓解熵崩溃问题。 |
| **Dynamic Sampling** | 动态筛选训练样本，剔除那些“全对”或“全错”的 prompt。 | 提高样本效率与梯度信号质量。 |
| **Token-level Policy Gradient** | 在 token 级别而非整句级别进行策略梯度优化。 | 适应长链式推理任务，缓解稀疏奖励问题。 |
| **Overlong Reward Shaping** | 对长生成序列的奖励进行平滑或延迟处理。 | 提升训练稳定性，减少奖励震荡。 |

---

## 三、数学形式（简化）

设旧策略为 `pi_old`，新策略为 `pi_theta`。  
定义重要性比率：

$$r_t = \frac{\pi_{\theta}(a_t | s, a_{<t})}{\pi_{\text{old}}(a_t | s, a_{<t})}$$

DAPO 的优化目标（简化形式）：

$$L(\theta) = \mathbb{E}\left[\min\big(r_t A_t,\, \text{clip}(r_t,\, 1 - \epsilon_{\text{low}},\, c_{\text{upper}}) A_t\big)\right]$$

其中 $A_t$ 为优势项，$c_{\text{upper}} > 1 + \epsilon$（比 PPO 的上限更高）。

训练时还会加入：
- token-level loss；
- reward shaping；
- 动态采样过滤策略。

---

## 四、算法流程（概要）

1. 从 SFT（监督微调）模型出发，生成若干响应样本；
2. 使用 **Dynamic Sampling** 筛选有效样本；
3. 计算每个样本或 token 的奖励 / 优势；
4. 应用 **Clip-Higher** 策略做策略梯度更新；
5. 利用 token-level loss 和 Overlong Reward Shaping 稳定训练。

---

## 五、优点与限制

### ✅ 优点
- 专为 **长链推理（Chain-of-Thought）** 任务设计；
- 提供细粒度 token 级优化；
- 改善样本效率与稳定性；
- 开源、可复现；
- 在数学推理任务中显著优于传统 PPO/GRPO。

### ⚠️ 限制
- 仍需大规模算力（例如 32B 模型）；
- 任务依赖明确可验证奖励（如数学题判定）；
- RL 训练仍需谨慎调参（clip 上限、采样策略等）；
- 目前验证主要集中于数学/逻辑推理任务。

---

## 六、适用场景与实践建议

**适用场景：**
- 多步推理任务（数学、逻辑、程序验证）
- 长文本生成与自我纠错任务
- 可计算奖励的 RL 场景（reward from correctness / verifier）

**实践建议：**
- 确保奖励函数可自动化计算；
- 在小模型上调 clip、token-loss 权重等参数；
- 监控策略熵、奖励趋势防止崩溃；
- 使用动态采样提升样本有效性；
- 资源受限时先模拟后扩展。

---

## 七、与其他算法对比

| 算法 | 核心思想 | 优点 | 局限 |
|------|-----------|------|------|
| **PPO** | 重要性采样 + 裁剪 | 稳定、通用 | 对 LLM 稀疏奖励不适用 |
| **GRPO** | Grouped Reward PPO（组奖励） | 改善 group-based 样本效率 | 熵崩溃问题明显 |
| **DAPO** | Clip-Higher + Dynamic Sampling + Token-level Loss | 稳定、细粒度、可复现 | 依赖明确奖励信号 |

---

## 八、参考资源
- [DAPO 论文 (arXiv:2503.14476)](https://arxiv.org/abs/2503.14476)  
- [DAPO 官方项目页面](https://dapo-sia.github.io/)  
- [技术解读 - Deep Dive into Open-Source RL for LLMs](https://adasci.org/deep-dive-into-open-source-rl-for-large-scale-llms-dapo/)  

---

> **简要总结：**  
> DAPO 是一种专为 LLM 推理任务设计的强化学习优化算法，核心思想是通过「更高的 clip 上限」「动态采样」「token 级梯度」「奖励平滑」来提高训练稳定性与样本效率。  
> 它可视为对 GRPO/PPO 的系统性改进与开源化实现。
