
# Always-on 移动多模态嵌入 (Mobile Multimodal Embedding)

-----

## 目录

  - [环境安装](https://www.google.com/search?q=%23%E7%8E%AF%E5%A2%83%E5%AE%89%E8%A3%85)
      - [ImageBind](https://www.google.com/search?q=%23ImageBind)
      - [ImageBind Trainer with LoRA fine-tuning](https://www.google.com/search?q=%23ImageBind-Trainer-with-LoRA-fine-tuning)
  - [代码结构](https://www.google.com/search?q=%23%E4%BB%A3%E7%A0%81%E7%BB%93%E6%9E%84)
  - [用法](https://www.google.com/search?q=%23%E7%94%A8%E6%B3%95)
      - [搜索](https://www.google.com/search?q=%23%E6%90%9C%E7%B4%A2)
      - [微调](https://www.google.com/search?q=%23%E5%BE%AE%E8%B0%83)
      - [推理](https://www.google.com/search?q=%23%E6%8E%A8%E7%90%86)
  - [train\_lumen\_imagenet.py](https://www.google.com/search?q=%23train_lumen_imagenet.py)

-----

## 环境安装

### ImageBind

1.  首先，克隆仓库。
    ```
    git clone
    ```
2.  安装 PyTorch 1.13+ 和其他第三方依赖。
    ```
    conda create --name imagebind python=3.8 -y
    conda activate imagebind
    pip install -r requirements.txt
    ```
3.  在使用 `train.py` 脚本且不带 `--headless` 参数时，请安装 `matplotlib`。
4.  更多信息，请遵循 [原始说明](https://github.com/facebookresearch/ImageBind)。

-----

## 代码结构

所有绘图脚本都放在 `plot_scripts/` 目录下。

```
├── api/
│   └── *数据预处理器*
├── datasets/
│   └── *数据加载器*
├── ImageBind-LoRA/
│   └── *lora 训练代码*
├── lightning_logs/
│   └── *自动生成的日志*
├── logs/
│   └── *手动生成的日志*
├── metrics/
│   └── *评估指标和基准脚本*
├── models/
│   └── *神经网络模型的定义和执行流程*
├── plot_scripts/
│   └── *绘制图表*
├── audio_process.py
├── data.py
├── lumen_2_infer.py
├── lumen_imagenet.py
├── ...
└── train.py
```

-----

## 用法

### 搜索

###### 暂时不会分层加载，搜索代码先搁置
在 `search.py` 中，你可以找到一个使用模型搜索目标标签图像的示例。

  - 要尝试 **LoRA 微调模型**，请将 `lora` 设置为 `True`，并在此脚本中设置微调模型的路径 `lora_dir` 和 `LoRA.apply_lora_modality_trunks()` 中的参数。
  - 要尝试 **原始 ImageBind 模型**，请将 `lora` 设置为 `False`。

**示例说明：**
**Imagenet** 数据集包含 1000 个类别，搜索对应 "stingray", "cock", "hen" 这三个词的图像，并获取我们搜索到的图像的精确率和召回率。

### 微调

修改 `train.py` 以适应 ImageNet 数据集的训练，修改后的代码存储在 `train_imagenet.py` 中。

以下是关于 `train.py` 的信息。要训练模型，请运行：

```
python train.py --batch_size 12 --max_epochs 500 \
        --lora --lora_modality_names vision text \
        --self_contrast --datasets dreambooth
```

你可以通过设置 `--loggers` 参数来启用 `comet`、`wandb` 或 `tensorboard` 进行日志记录。确保事先安装了相应的日志包以及必要的环境变量。

要指定对哪些层或模态应用 LoRA，请使用 `--lora_layer_idxs` 和 `--lora_modality_names` 参数。要覆盖某个特定模态的层数，你可以专门指定该模态，例如，添加以下参数来为视觉主干的前 6 层指定 LoRA：

```
--lora_layer_idxs_vision 1 2 3 4 5 6
```

要在 GPU 上训练（目前只支持单 GPU，多 GPU 训练即将添加），请设置 `--device` 参数：

```
--device cuda:0
```

`example.py` 中使用的 LoRA 模型（检查点位于 `.checkpoints/lora/550_epochs/`，后缀为 `_dreambooth_last.safetensors`）在 **12 GB VRAM 的 3080Ti** 上训练了约 2 小时，消耗了 5.66 GB VRAM 和约 4 GB RAM。该模型在不到 30 分钟内收敛到了相似的状态。

**信息：**

  * 8.0 M 可训练参数
  * 1.2 B 不可训练参数
  * 1.2 B 总参数
  * 4,815.707 总模型估计参数大小 (MB)

我们将训练参数设置为：

```
# 安装 comet-ml:
#       pip install comet-ml
# 并设置环境变量:
#       export COMET_API_KEY=<MY_API_KEY>
#       export COMET_WORKSPACE=<MY_WORKSPACE_NAME>
#       export COMET_PROJECT_NAME=Imagebind-lora
python train.py --batch_size 12 --max_epochs 550 --num_workers 4 \
                --lora --lora_modality_names vision text \
                --self_contrast --datasets dreambooth \
                --device cuda:0 --headless --loggers comet
```

**注意：**
要执行线性探测（仅优化每个模态头部中的最后一层），请保留所有参数，将 `--lora` 替换为 `--linear_probing`（两者不能在同一次运行中同时设置）。在接下来的训练会话中运行 `--lora` 时，检查点会自动加载并保存，前提是 `--lora_checkpoint_dir` 保持不变。

### 推理

在 `test_imagenet.py` 中，你可以找到一个在 ImageNet 数据集上使用模型进行推理的示例。

  - 要尝试 **LoRA 微调模型**，请将 `lora` 设置为 `True`，并在此脚本中设置微调模型的路径 `lora_dir` 和 `LoRA.apply_lora_modality_trunks()` 中的参数。
  - 要尝试 **原始 ImageBind 模型**，请将 `lora` 设置为 `False`。
    你还可以使用 `imagebind_model.imagebind_huge()` 设置每个模态的主干块。

-----

## train\_lumen\_imagenet.py

用于训练 lumen 模型。

使用以下代码训练 experiment6：

```
self.model = lumen6_model.imagebind_huge(pretrained=True,vision_num_blocks_1=1,vision_num_blocks_2=30,text_num_blocks=24)
```

使用以下代码训练 experiment1：

```
self.model = lumen_model.imagebind_huge(pretrained=True,vision_num_blocks_1=1,vision_num_blocks_2=30,text_num_blocks=24)
```

其他参数（大部分可以在 “parser” 函数中更改）：

  - `lora_dir`:
    ```
    parser.add_argument("--lora_checkpoint_dir", type=str, default="./.checkpoints/lora/lume_lora-666!!!")
    ```
  - `device`:
    ```
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training and validation")
    ```
  - ....
    ```
    trainer = Trainer(accelerator="gpu" if "cuda" in device_name else "cpu", devices=[1,2,3], deterministic=True)
    ```

-----
