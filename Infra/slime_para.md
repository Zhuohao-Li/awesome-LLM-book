# slime parameters

refer to `slime/utils/arguments.py`

```

==== Slime 参数汇总 ====
注：本文件还会通过 add_sglang_arguments(parser) 注入额外 SGLang 参数，未在此处列出。

--- Cluster ---
- --actor-num-nodes: 训练 actor 的节点数; default=1; type=int
- --actor-num-gpus-per-node: 每个节点用于训练 actor 的 GPU 数; default=8; type=int
- --critic-num-nodes: 训练 critic 的节点数; default=None; type=int
- --critic-num-gpus-per-node: 每个节点用于训练 critic 的 GPU 数; default=None; type=int
- --rollout-num-gpus: 推理使用的 GPU 总数（在 --colocate 下被忽略并设为 actor_num_gpus_per_node * actor_num_nodes）; default=None; type=int
- --rollout-num-gpus-per-engine: 每个推理引擎使用的 GPU 数（类似 sglang 的 tp_size）; default=1; type=int
- --num-gpus-per-node: rollout 每节点 GPU 数（colocate 下小于 8 需显式设置）; default=8; type=int
- --colocate: 推理引擎与 actor 同机部署（开启也会强制 --offload=True）; default=False; action=store_true
- --offload: 训练时将 rollout 生成器与 actor 下放到 CPU（colocate 时始终为 True）; default=False; action=store_true
- --distributed-backend: 分布式后端; default="nccl"; type=str
- --distributed-timeout-minutes: 分布式超时（分钟）; default=10; type=int

--- Train ---
- --train-backend: 训练后端; default="megatron"; type=str; choices=["megatron","fsdp"]
- --true-on-policy-mode: 启用 true-on-policy 模式; default=False; action=store_true

--- Rollout ---
- --hf-checkpoint: 用于初始化 sglang 与 tokenizer 的 HF checkpoint; default=None; type=str
- --use-hf-config-for-megatron: 使用 HF 配置定义 Megatron 架构; action=store_true
- --model-name: 将 Megatron 权重转换为 HF 时使用的模型名; default=None; type=str
- --rollout-function-path: rollout 生成函数路径; default="slime.rollout.sglang_rollout.generate_rollout"; type=str
- --rollout-temperature: 推理温度; default=1.0; type=float
- --rollout-top-p: 推理 top-p; default=1.0; type=float
- --rollout-top-k: 推理 top-k; default=-1; type=int
- --rollout-max-context-len: 推理最大上下文长度; default=None; type=int
- --rollout-max-prompt-len: 推理最大 prompt 长度; default=None; type=int
- --rollout-max-response-len: 推理最大输出长度（sglang 的 max_tokens）; default=1024; type=int
- --rollout-skip-special-tokens: 推理时跳过特殊 token; default=False; action=store_true
- --rollout-stop: 推理停止词; default=None; type=str; nargs="+"
- --rollout-stop-token-ids: 推理停止 token id; default=None; type=int; nargs="+"
- --rollout-shuffle: rollout 时是否打乱 prompts; default=False; action=store_true
- --rollout-seed: rollout 随机种子; default=42; type=int
- --over-sampling-batch-size: 采样批粒度（缺省使用 rollout_batch_size）; default=None; type=int
- --dynamic-sampling-filter-path: 动态采样过滤函数路径; default=None; type=str
- --partial-rollout: 启用部分 rollout，未完成样本回收到 buffer; default=False; action=store_true
- --custom-generate-function-path: 仅替换示例中的 generate() 实现; default=None; type=str
- --buffer-filter-path: buffer 选择函数路径; default=None; type=str
- --update-weight-buffer-size: 分片更新权重的缓冲区大小（字节）; default=512 * 1024**2; type=int
- --update-weights-interval: 权重更新间隔; default=1; type=int
- --keep-old-actor: 训练过程中保留 rollout 模型; action=store_true
- --rollout-data-postprocess-path: 汇总 rollout 数据后调用的后处理函数路径; default=None; type=str
- --rollout-external: 使用外部 SGLang 实例; default=False; action=store_true
- --rollout-external-engine-addrs: 外部引擎地址列表; default=None; type=str; nargs="+"

--- Fault Tolerance ---
- --use-fault-tolerance: 启用 rollout 容错功能; default=False; action=store_true
- --rollout-health-check-interval: /health_generate 健康检查间隔（秒）; default=30.0; type=float
- --rollout-health-check-timeout: /health_generate 健康检查超时（秒）; default=30.0; type=float
- --rollout-health-check-first-wait: 初次健康检查前等待（秒）; default=300.0; type=float

--- Data ---
- --num-rollout: rollout 步数; default=None; type=int
- --num-epoch: 训练轮数（用于推导 num_rollout）; default=None; type=int
- --disable-rollout-global-dataset: 关闭全局数据集（dest=rollout_global_dataset False）; action=store_false
- --prompt-data: prompt 数据路径（jsonl）; default=None; type=str
- --apply-chat-template: 应用 chat 模板; default=False; action=store_true
- --input-key: JSON 数据键; default="input"; type=str
- --label-key: JSON 数据键; default=None; type=str
- --multimodal-keys: 多模态类型到数据键的 JSON 映射; default=None; type=json.loads
- --metadata-key: JSON 数据键; default="metadata"; type=str
- --tool-key: 数据集中工具字段的键; default=None; type=str
- --start-rollout-id: 起始 rollout 步; default=None; type=int
- --rollout-batch-size: 每次 rollout 的 prompt 数; required=True; type=int
- --n-samples-per-prompt: 每个 prompt 的生成数; default=1; type=int
- --global-batch-size: 全局 batch size（重置默认）; default=None; type=int
- --num-steps-per-rollout: 每个 rollout 的训练步数; default=None; type=int
- --micro-batch-size: micro batch size（重置默认）; default=1; type=int
- --balance-data: 在数据并行维度平衡 token 数; default=False; action=store_true
- --use-dynamic-batch-size: 启用动态 batch size; default=False; action=store_true
- --max-tokens-per-gpu: 动态 batch 的单卡最大 token 数; default=None; type=int
- --log-probs-max-tokens-per-gpu: 计算 log probs 的单卡最大 token 数; default=None; type=int

--- Eval ---
- --eval-function-path: eval 生成函数路径（缺省沿用 rollout 路径）; default=None; type=str
- --eval-interval: 评估间隔（重置默认）; default=None; type=int
- --eval-prompt-data: 评估数据集名与路径成对传入; default=None; type=str; nargs="+"
- --eval-input-key: JSON 数据键; default=None; type=str
- --eval-label-key: JSON 数据键; default=None; type=str
- --eval-tool-key: JSON 数据键; default=None; type=str
- --n-samples-per-eval-prompt: 每个评估 prompt 的生成数; default=1; type=int
- --eval-temperature: 评估温度; default=None; type=float
- --eval-top-p: 评估 top-p; default=None; type=float
- --eval-top-k: 评估 top-k; default=None; type=int
- --eval-max-response-len: 评估最大输出长度; default=None; type=int
- --eval-min-new-tokens: 评估最小新增 token 数; default=None; type=int

--- Algo ---
- --ref-load: 参考模型 checkpoint; default=None; type=str
- --ref-ckpt-step: 参考模型 checkpoint step; default=None; type=int
- --load: 训练模型加载路径（重置默认）; default=None; type=str
- --save: 训练模型保存路径（重置默认）; default=None; type=str
- --save-interval: 保存间隔（重置默认）; default=None; type=int
- --seed: 随机种子（重置默认）; default=1234; type=int
- --clip-grad: 梯度裁剪（重置默认）; default=1.0; type=float
- --calculate-per-token-loss: 逐 token 计算 loss（重置默认）; action=store_true
- --lr: 学习率（重置默认）; default=1e-6; type=float
- --num-critic-only-steps: 仅 critic 的训练步数; default=0; type=int
- --critic-load: critic 模型加载路径; default=None; type=str
- --critic-save: critic 模型保存路径; default=None; type=str
- --critic-lr: critic 学习率; default=None; type=float
- --critic-lr-warmup-iters: critic 线性 warmup 迭代数; default=0; type=int
- --eps-clip: PPO clip 范围; default=0.2; type=float
- --eps-clip-high: PPO clip 上界; default=None; type=float
- --eps-clip-c: Dual-clip PPO 的下界; default=None; type=float
- --value-clip: value loss 的 clip; default=0.2; type=float
- --kl-coef: 奖励整形的 KL 系数（用于优势前）; default=0.00; type=float
- --loss-type: 损失函数类型; default="policy_loss"; type=str; choices=["policy_loss","sft_loss","custom_loss"]
- --custom-loss-function-path: 自定义损失函数路径; default=None; type=str
- --kl-loss-type: KL 损失类型; default="k1"; type=str; choices=["k1","k2","k3","low_var_kl"]
- --advantage-estimator: 优势估计器; default="grpo"; type=str; choices=["grpo","gspo","reinforce_plus_plus","reinforce_plus_plus_baseline","ppo"]
- --disable-compute-advantages-and-returns: 关闭优势/回报计算（dest=compute_advantages_and_returns False）; action=store_false
- --use-kl-loss: 启用 GRPO 的 KL loss; default=False; action=store_true
- --kl-loss-coef: 最终 loss 中 KL 系数; default=0.0; type=float
- --ref-update-interval: 参考模型更新间隔（单位：rollout 步）; default=None; type=int
- --entropy-coef: 熵损系数; default=0.0; type=float
- --gamma: PPO GAE gamma; default=1.0; type=float
- --lambd: PPO GAE lambda; default=1.0; type=float
- --normalize-advantages: 归一化优势; default=False; action=store_true
- --disable-grpo-std-normalization: 关闭 GRPO 标准差归一化（dest=grpo_std_normalization False）; action=store_false
- --disable-rewards-normalization: 关闭奖励归一化（dest=rewards_normalization False）; action=store_false
- --use-rollout-entropy: 计算 actor/ref 的对数概率熵; default=False; action=store_true
- --use-tis: 启用 TIS（离策略重要性采样）; default=False; action=store_true
- --tis-clip: IS 比例裁剪阈值 C; default=2.0; type=float
- --tis-clip-low: IS 比例下界裁剪阈值 C; default=0; type=float
- --custom-tis-function-path: 自定义 TIS 函数路径; default=None; type=str
- --use-routing-replay: 启用 routing replay; default=False; action=store_true

--- Router ---
- --use-slime-router: 使用 SlimeRouter 文本路由（替代 SGLang token 路由）; default=False; action=store_true
- --slime-router-middleware-paths: 中间件路径; default=""; type=str; nargs="+"

--- W&B ---
- --use-wandb: 启用 wandb; default=False; action=store_true
- --wandb-mode: 运行模式; default=None; type=str; choices=["online","offline","disabled"]
- --wandb-dir: wandb 日志目录; default=None; type=str
- --wandb-key: API key; default=None; type=str
- --wandb-host: 主机; default=None; type=str
- --wandb-team: 团队; default=None; type=str
- --wandb-group: 分组; default=None; type=str
- --wandb-project: 项目名（重置默认）; default=None; type=str
- --disable-wandb-random-suffix: 关闭随机后缀（dest=wandb_random_suffix False）; default=True; action=store_false
- --wandb-always-use-train-step: 总是使用训练 step 作为指标步; default=False; action=store_true
- --log-multi-turn: 记录多轮 rollout 信息; default=False; action=store_true
- --log-passrate: 记录 pass@n; default=False; action=store_true
- --log-reward-category: 记录奖励类别统计（指定奖励字典键）; default=None; type=str
- --wandb-run-id: 运行 ID; default=None; type=str

--- TensorBoard ---
- --use-tensorboard: 启用 TensorBoard; default=False; action=store_true
- --tb-project-name: TB 项目名; default=None; type=str
- --tb-experiment-name: TB 实验名; default=None; type=str

--- Debug ---
- --save-debug-rollout-data: 保存 rollout 调试数据路径模板; default=None; type=str
- --load-debug-rollout-data: 加载 rollout 调试数据路径模板; default=None; type=str
- --debug-rollout-only: 仅运行 rollout（不训练）; default=False; action=store_true
- --debug-train-only: 仅训练（不启 SGLang 服务器）; default=False; action=store_true
- --save-debug-train-data: 保存训练调试数据路径模板; default=None; type=str
- --dump-details: 导出训练细节; default=None; type=str
- --memory-snapshot-dir: 内存快照目录; default="."; type=str

--- Network ---
- --http-proxy: HTTP 代理; default=None; type=str
- --use-distributed-post: 启用分布式 POST; default=False; action=store_true

--- Reward Model ---
- --rm-type: 奖励模型类型; default=None; type=str
- --reward-key: 从奖励字典中取值的键; default=None; type=str
- --eval-reward-key: 奖励键的评估版本; default=None; type=str
- --group-rm: 对整组做 RM; default=False; action=store_true
- --rm-url: 远程 RM 服务 URL（remote_rm）; default=None; type=str
- --custom-rm-path: 自定义奖励模型函数路径; default=None; type=str
- --custom-reward-post-process-path: 奖励后处理函数路径; default=None; type=str

--- Rollout Buffer ---
- --rollout-buffer-url: Rollout buffer 服务 URL; default=None; type=str
- --fetch-trajectory-retry-times: 取样轨迹重试次数（-1 无限重试）; default=-1; type=int
- --min-batch-collection-ratio: 最小批采集比例; default=1; type=float
- --rollout-task-type: 任务类型; default="math"; type=str
- --loss-mask-type: loss mask 类型; default="qwen"; type=str; choices=["qwen","qwen3","distill_qwen"]

--- Custom Megatron Plugins ---
- --custom-megatron-init-path: 自定义 Megatron 初始化钩子; default=None; type=str
- --custom-megatron-before-log-prob-hook-path: 计算 log prob 前的钩子; default=None; type=str
- --custom-megatron-before-train-step-hook-path: 训练步前的钩子; default=None; type=str

--- CI ---
- --ci-test: 启用 CI 测试; action=store_true
- --ci-disable-kl-checker: 关闭 KL 检查器; action=store_true
- --ci-metric-checker-key: 指标检查键; default=None; type=str
- --ci-metric-checker-threshold: 指标阈值; default=None; type=float

--- Other ---
- --custom-config-path: 自定义函数参数的 YAML 配置路径; default=None; type=str
- --padded-vocab-size: 词表补齐大小; default=None; type=int

```