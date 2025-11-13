### slime：是一个高性能，易用，解耦的RL框架，他支持了GLM系列 flagship 模型的RL，GLM是zhipu的系列模型，GLM-4.5（355B-A32B），在stem reasoniong上效果拔群

- 解耦：原生Megatron+SGLang+monkey patch / Data Buffer，提供了方便的rollout接口，用户只需要做customize rollout data generation的设计。同时提供了多样的RL env和RM，multi-agent system等复杂RL要求也可以胜任
    - monkey patch举个例子就是offload nccl时模仿cuda VMM （virtual memory management）API的接口设计
    - 解耦的意思是，把SGL通过sgl-router API暴露出来，用户可以通过API自定义如何与SGL交互，data buffer cache所有rollout数据，训练数据，并且在train/rollout直接进行dispatch
    
- 高性能：支持全量训推优化，包括但不限于Megatron（5D 并行，量化，CPU adam），SGLang高吞吐低延时（FP8 rollout，DeepEP，MTP），RL框架设计（Megatron parameter offload -> more KV cache，model参数更新优化），算法（GSPO，TIS，true on policy等）
    - 单条数据的decode速度决定了RL的训练速度上限。这个和Pretrain/SFT/CPT通过加卡就一定可以加速的scaling law不同，inference加速无法通过加卡来scale。因此，slime可以原生开启SGL的所有加速方法：（1）量化（2）speculative decoding（3）MTP，对于大MoE，也支持（4）DeepEP等通信库。我之前给SGL提过一个PR #2627，是关于利用flux和TE这种fused大kernel来做推理的计算通信overlap，针对Dense的MLP的，主要是AG/RS和gemm的overlap。当时MoE还没有这么火，后来有了DeepEP这样的通信库，让通信计算overlap在MoE上的效果更加显著。这些优化可以大幅提高inference speed，GLM4.5从10t/s到70t/s
    - offload megatron：只要 kv cache 不溢出，推理 batch size 的提升并不会明显影响训练的延时。这里 kv cache 溢出是指推理过程中，当数据的回复长度都很长时，kv cache 空间不够，就需要把某些生成到一半的数据先踢出队列，等其他数据推理完，腾出 kv cache 的空间，再重新进行 prefill 和后续的推理。如果一条回复长度为 64k 的数据在推理过程中等待了其他数据 decode 32k token，相当于他的总时长对应了 decode 96k token，这对 RL 训练速度有很大影响。所以我们需要根据 rollout bsz，resp len，mem-frac计算出kv cache不满的情况下需要的最少GPU数量。
        - cpu adam
        - tms使用了类似VMM API来管理显存，类似OS的虚拟地址和物理地址，我们可以管理这些CUDA IPC Handler来实现offload Megatron的GPU tensor。对于MoE model，除开tensor，还会放置nccl buffer，带来不小的显存残留。
    - 参数更新
        - 从CUDA IPC handler -> 异步通信 -> 组bucket减少频繁http API调用 -> flatten tensor组大size避免open close hanlder开销 -> cache param和expert mapping等参数
- 一个好的RL框架：
    - 是否支持大MoE训练
    - 是否支持SGL mem frac调至0.7以上，保证足够的kv cache
    - 是否支持FP8或更低的推理
    - 是否支持DeepEP
    - 是否支持speculative decoding
    - 是否支持高效训练backend

我做的：
- FSDP，支持更灵活的FSDP训练框架
    - reduce memory overhead for model weight update in colocated #357，主要就是一些之前讲到的colocate下参数更新的招数
    - 训推分离下UpdateWeightFromDistributed #341
    - 验证FSDP install #302
    - EP(in progress),CP

- Onpolicy：支持更多onpolicy的修正算法，包括：
    - TIS （train-infer mismatch）
    - True on Policy Kernel （Batch Invarient Kernel）

- Agentic RL
    - 提供了更完备的search r1 with slime recipe （#688）
    - 包括使用Qwen3训练，Local Dense Retriever，TIS

### Accio：这是一个business agent，我负责mid train和post train
- Data
    - 网关数据清洗，不同LLM call API调用各式（OAI SDK，Chat Completion，Generate，Gemini等），Tool-use调用格式处理，不同文件格式处理（json，parquet，csv），统一sharegpt格式。收集清洗RL env团队提供的tool info（customize search/retrieval，内部downstream tool
    - SFT数据生成：filter/cluster/联想/reject sampling。主要就是用hirarchical clustering的思想，用prompt template检查prompt prefix判断意图，把prompts都group起来。LLM再来识别意图，并且enrich做reject sampling。同时balence distribution
    - RL数据生成：人来标注+synthesis。用LLM识别相关性，通过info graph组合问题，生成答案，人为验证，人工标注。
- RL/Infra
    - CPT/SFT我们都采用内部定制的Megatron（Damo-AGI/Qwen）。这一部分比较fix了，主要给Qwen系列模型提供高效、灵活的训练支持。flash-moe. Qwen3系列包含多个 size 的 dense 和 MoE 模型，其中 Dense 模型包括：0.6B、1.7B、4B、8B、14B 和 32B，MoE 模型包括 15B-A2B（内部使用不开源）、30B-A3B 和 235B-A22B。内部提供的 ckpt 比外部开源更多，其中包括多个阶段的预训练模型，如 S1 （不开源）、S2（不开源）、Base（Dense 模型开源，MoE 模型不开源）。它们分别为不同的预训练阶段，其中 S1 为第一阶段（约 30 T tokens），S2 为 第二阶段（STEM 和 Coding 占大部分，约 5 T）、Base（退火阶段，打分器筛选的高质量数据以及长序列数据为主，约 200B）。如无 S1，即该模型不存在 S1 阶段，这是因为部分模型是使用蒸馏以及 upscaling 得到的原因。
    - 根据我们的需求搭建了RL infra框架（flexible的rollout data generation：带着agent一起训练），出现的问题是：（1）长尾：partial rollout（2）prefill很长：chunked prefill（3）训推分离存在bubble：异步或者colocate（4）复杂的env（tool-use，RM）：分离serve走RDMA网络接口
    - RM：LLM judger + rubrics/checklist

- Agent
    - Deep Research：planning/react/context management

- Research：
    - 
