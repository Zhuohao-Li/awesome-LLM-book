# interview

### nccl

find nccl version
```
find /usr/ -name "libnccl*"
```

### nvlink
```
# check topo
nvidia-smi topo -m

# check nvlink status
nvidia-smi nvlink --status
```

### pre-norm & post-norm


以单个子层为例（比如 Self-Attention），设输入为 \( x_l \)，子层函数为 \( F(\cdot) \)。

### 🔸 Post-Norm（原版 Transformer, Vaswani 2017）

$$
y_l = x_l + F(x_l)
$$

$$
x_{l+1} = \text{LayerNorm}(y_l)
$$

### 🔹 Pre-Norm（现代 LLM）

$$
y_l = x_l + F(\text{LayerNorm}(x_l))
$$

$$
x_{l+1} = y_l
$$

差别：归一化的位置不同。


### 🔸 Post-Norm 的梯度路径

反向传播链式法则：

$$
\frac{\partial L}{\partial x_l}
  = \frac{\partial L}{\partial x_{l+1}}
     \cdot
     \frac{\partial x_{l+1}}{\partial y_l}
     \cdot
     \frac{\partial y_l}{\partial x_l}
$$

代入结构：

$$
x_{l+1} = \text{LN}(x_l + F(x_l))
$$

得到：

$$
\frac{\partial x_{l+1}}{\partial x_l}
= J_{\text{LN}}(x_l + F(x_l)) \cdot
  \left(I + \frac{\partial F(x_l)}{\partial x_l}\right)
$$

其中 \( J_{\text{LN}} \) 表示 LayerNorm 的 Jacobian。

**问题：**

- LayerNorm 的 Jacobian 不是恒等映射；
- 每一层的梯度都要乘上 \( J_{\text{LN}} \)；
- 当层数很多时，这些非恒等矩阵连乘，数值可能指数式衰减或放大；
- 因此梯度容易消失或爆炸。

换句话说，残差的“恒等路径”被 LayerNorm 打断了，梯度无法直接顺着残差通道流回前面层。


### 🔹 Pre-Norm 的梯度路径

现在看 Pre-Norm：

$$
y_l = x_l + F(\text{LN}(x_l))
$$

反向传播：

$$
\frac{\partial L}{\partial x_l}
  = \frac{\partial L}{\partial y_l}
     \cdot
     \left[
       I + 
       \frac{\partial F(\text{LN}(x_l))}{\partial \text{LN}(x_l)} 
       \cdot 
       \frac{\partial \text{LN}(x_l)}{\partial x_l}
     \right]
$$

关键在于：

$$
\frac{\partial L}{\partial y_l}
$$

可以直接沿着残差路径传递（那条 \( I \) 通道），即使 \( F(\cdot) \) 训练初期还不稳定。

这个 “+I” 项保证了：  
即使子层的梯度出问题，恒等残差通路仍然提供稳定的梯度回流路径。


### ✅ 结论

- 梯度能稳定地从输出传回输入；
- 深层网络（上百层）仍能收敛；
- 梯度不易爆炸或消失。


## FA

v1：基于「IO-aware」思想，通过瓦片 (tiling) 将 QKᵀ·V 计算中对慢速 HBM 的读写降到最少，从而提升速度与内存效率。
arXiv
+1

v2：改进线程块／warp 内任务划分、减少非‐矩阵乘 (non-GEMM) FLOPs、并行化单头 attention，以大幅提升 GPU 占用率与吞吐率。
arXiv
+1

v3：针对 Hopper／H100 架构，引入低精度 (FP8)、异步执行 (重叠 GEMM 与 softmax) 以及硬件专用指令 (TMA/WGMMA) 来进一步提升效率。
tridao.me
+1

v4：通过更细粒度的 warp 专化与流水线 (warp specialization + async pipeline)，软件模拟指数运算、数值复校 (correction step)，以在最新 Blackwell／SM10 架构上突破 PFLOPS 级别。


## GPU

* warp 是32个thread集合
* block 是多个warp集合，被SM调度
* grid 是所有block的集合

warp divergence会带来一个warp里的不同执行路径，导致性能损失，所以需要warp level的优化：
* warp shuffle/reduction：warp 内线程通过寄存器直接交换数据，不用写到共享内存
* warp specialization：不同 warp 执行不同角色（如加载数据 / 计算 / 写回）形成流水线
* warp occupancy：通过控制 warp 数量，最大化 SM 利用率

