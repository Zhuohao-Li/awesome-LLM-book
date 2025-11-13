# BCE and BCE logits weight

BCE Loss(Binary Cross-Entropy Loss)

$$
L = - [y \text{log}(\hat y) + (1-y) \text{log} (1-\hat y)]
$$

when groundtruth label $ y = 1 $, we only have

$$
L = - \text{log} ( \hat y)
$$

so we hope predicted $\hat y$ is more closed to 1.

when groundtruth label $ y = 0 $, we only have

$$
L = - \text{log} ( 1 - \hat y)
$$

so we hope predicted $\hat y$ is more closed to 0.

For vector like process:

$$
\text{BCE} = - \cfrac{1}{N} \sum ^{N} _{i=1}[y_i \text{log}(\hat y_i) + (1-y_i) \text{log} (1-\hat y_i)]
$$

```
import torch
import torch.nn as nn

loss_fn = nn.BCELoss()
# 如果模型输出是 logit（未经过 sigmoid）
# 应使用：
loss_fn = nn.BCEWithLogitsLoss()
```

It is used in binary classification.

BCEWithLogitsLoss is more stable than BCE, it changed the inpout from prob to logitrs


$$
L = - [y \text{log}(\sigma (x)) + (1-y) \text{log} (1- \sigma (x) )]
$$

when $x$ is very large, $ \sigma (x) \rightarrow 1  $, $\text{log}(1 - \sigma (x)) \rightarrow -\infin$

# Temperature in LLM inference

$$
P(y_i) = \frac{e^{z_i/T}}{\sum _j e^{z_j/T}}
$$

$T=1$ means origin distribution, $T>1$ means more smooth, during eval, $T=0$ for deterministic inference.