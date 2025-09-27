Of course, I can translate and organize this README for you. Here is the Chinese version in the original Markdown format.

-----

# Always-on 移动多模态嵌入

-----

## 目录

  - [环境安装](https://www.google.com/search?q=%23%E7%8E%AF%E5%A2%83%E5%AE%89%E8%A3%85)
  - [代码结构](https://www.google.com/search?q=%23%E4%BB%A3%E7%A0%81%E7%BB%93%E6%9E%84)
  - [使用方法](https://www.google.com/search?q=%23%E4%BD%BF%E7%94%A8%E6%96%B9%E6%B3%95)
      - [搜索演示](https://www.google.com/search?q=%23%E6%90%9C%E7%B4%A2%E6%BC%94%E7%A4%BA)
      - [微调](https://www.google.com/search?q=%23%E5%BE%AE%E8%B0%83)
  - [端到端系统工作流](https://www.google.com/search?q=%23%E7%AB%AF%E5%88%B0%E7%AB%AF%E7%B3%BB%E7%BB%9F%E5%B7%A5%E4%BD%9C%E6%B5%81)

-----

## 环境安装 [典型安装时间：10 分钟]

请确保递归克隆此仓库以包含子模块：

```
git clone --recurse-submodules -j8 https://github.com/fabawi/ImageBind-LoRA.git
```

有关安装，请遵循[原始使用说明](https://github.com/facebookresearch/ImageBind)。在使用 `train.py` 脚本且不带 `--headless` 参数时，请安装 `matplotlib`。

**警告：** 如果您收到此错误 -\> `'FastAPI' object has no attribute 'debug'`，请将 `fastapi` 升级到最新版本：

```
pip install --upgrade fastapi
```

安装 PyTorch 1.13+ 和其他第三方依赖：

```
conda create --name imagebind python=3.8 -y
conda activate imagebind
pip install -r requirements.txt
```

在使用 `train.py` 脚本且不带 `--headless` 参数时，请安装 `matplotlib`。
更多信息，请遵循[原始说明](https://github.com/facebookresearch/ImageBind)。

-----

## 代码结构

TODO: 将所有绘图脚本放入 `plot_scripts/` 目录下。

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
│   └── *绘制图表* ├── audio_process.py
├── data.py
├── lumen_2_infer.py
├── lumen_imagenet.py
├── ...
└── train.py
```

-----

## 使用方法

### 搜索演示

在 `search.py` 中，您可以找到一个使用模型搜索目标标签图像的示例。

  - 要尝试 **LoRA 微调模型**，请将 `lora` 设置为 `True`，并在脚本中设置微调模型的路径 `lora_dir` 和 `LoRA.apply_lora_modality_trunks()` 中的参数。
  - 要尝试 **原始 ImageBind 模型**，请将 `lora` 设置为 `False`。
    您还可以使用 `imagebind_model.imagebind_huge()` 设置每个模态的主干块。

**示例说明：**
Imagenet 数据集包含 1000 个类别，搜索对应 "stingray", "cock", "hen" 这三个词的图像，并获取我们搜索到的图像的精确率和召回率。

### 微调

修改 `train.py` 以适应 ImageNet 数据集的训练，修改后的代码存储在 `train_imagenet.py` 中。
以下是关于 `train.py` 的信息。要训练模型，请运行：

```
python train.py --batch_size 12 --max_epochs 500 \
        --lora --lora_modality_names vision text \
        --self_contrast --datasets dreambooth
```

您可以通过设置 `--loggers` 参数来启用 `comet`、`wandb` 或 `tensorboard` 进行日志记录。请确保事先安装相应的日志包以及必要的环境变量。

要指定对哪些层或模态应用 LoRA，请使用 `--lora_layer_idxs` 和 `--lora_modality_names` 参数。要覆盖某个特定模态的层数，您可以专门指定该模态，例如，添加以下参数来为视觉主干的前 6 层指定 LoRA：

```
--lora_layer_idxs_vision 1 2 3 4 5 6
```

要在 GPU 上训练（目前支持单个 GPU，多 GPU 训练即将添加），请设置 `--device` 参数：

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
# installed comet-ml:#       pip install comet-ml# and set the env variables:#       export COMET_API_KEY=<MY_API_KEY>#       export COMET_WORKSPACE=<MY_WORKSPACE_NAME>#       export COMET_PROJECT_NAME=Imagebind-lora

python train.py --batch_size 12 --max_epochs 550 --num_workers 4 \
                --lora --lora_modality_names vision text \
                --self_contrast --datasets dreambooth \
                --device cuda:0 --headless --loggers comet
```

**注意：**
要执行线性探测（仅优化每个模态头部中的最后一层），请保留所有参数，将 `--lora` 替换为 `--linear_probing`（两者不能在同一次运行中同时设置）。在接下来的训练会话中运行 `--lora` 时，检查点会自动加载和保存，前提是 `--lora_checkpoint_dir` 保持不变。

-----

## 端到端系统工作流

E2E（端到端）[预计运行时间：使用提供的检查点为 1 小时]

要构建一个端到端系统，您需要执行 4 个步骤。在 `run_dataset.py` 中，您可以看到 clotho 数据集的完整管道，同样的方法也适用于其他数据集。

### 步骤 1：使用不同模型层获取所有数据嵌入

在我们的技术中，我们需要动态地嵌入每个数据，因此需要准备不同模型层的嵌入。

在 `get_embedding_cltho.py` 中，我们可以计算 clotho 数据集在特定音频层的嵌入。因此在 `run_dataset.py` 的步骤 1 中，您需要迭代所有音频层（从 1 到 Imagebind 的完整音频层）。

`get_embedding_cltho.py` 的参数：

  * **输入：**
      * `--lora_layers`：'定义模型层'
      * `--lora_dir`：'lora 参数的路径'
      * `--embedding_dir`：'保存嵌入的路径'
      * `--dataset`：'数据集名称'
  * **输出：**
      * 相关数据集嵌入保存在 `embedding_dir` 中。

### 步骤 2：在不同模型层推断数据集，以获取数据预测结果。

此步骤的目标是获取数据集中每个单独数据的预测结果。例如，在音频层 = 7 时，单个数据的 R@N 输出结果可能为 0，而在层 = 9 时，它可能为 1。

在 `test_clotho_val.py` 中，我们计算了一个数据集在特定模型层的所有预测，结果为 0/1，'0' 表示错误，而 '1' 表示正确。在步骤 2 中，我们迭代从 1 到完整层的所有层，然后获取每个数据在不同层上的全部结果。

`test_clotho_val.py` 的参数：

  * **输入：**
      * `--audio_num_blocks`：'定义模型层'
      * `--lora_dir`：'lora 参数的路径'
      * `--embedding_path`：'保存嵌入的路径'
      * `--version`：'实验的标签，通常包含数据集名称和实验方法（例如，是否使用 lora，是否带有 lora head），用于识别文件'
  * **输出：**
      * 每个数据的预测结果保存在 txt 文件中，通常在 `'results/clotho_head/R{N}'`。

### 步骤 3：获取每个数据的最小层

使用步骤 2 中得到的 txt 文件来获取结果为 1 的最小层。在步骤 2 中，我们只得到了每个数据在每个模型层上的结果，但最终目标是获取检索每个数据所需的最小层。因此，我们需要找到结果为 '1' 的第一层，而 `get_layers_clotho.py` 这个 Python 文件可以完成这项工作。

`get_layers_clotho.py` 的参数：

  * **输入：**
      * 步骤 2 中得到的 txt 文件。
  * **输出：**
      * 每个数据所需的最小层保存在 txt 文件中，通常在 `'results/clotho_head/R{N}/layers.txt'`。

### 步骤 4：使用步骤 3 中得到的标签来训练预测模型

我们使用数据集在特定模型层 N 上的嵌入，以及每个数据所需的最小层（标签）来训练模型，让模型预测某个数据需要多少层。

`model_predict_lora.py` 的参数：

  * **输入：**
      * 每个数据的标签，通常在 `'results/clotho_head/R{N}/layers.txt'`。
  * **输出：**
      * 模型检查点。

### 步骤 5：使用步骤 4 中得到的模型动态嵌入数据集并获取预测结果，以选择 top Q 个结果进行细粒度搜索。

  * **输入：**
      * **N**：您希望输入模型的层数，通常 N 越大，准确度越高，但计算时间也越长。
      * **S**：与您输入模型的 R@S 标签中的 **S** 相同。S 越大，模型预测的层数越少，但预测模型的准确度越高。
      * **Q**：在动态搜索中获得的 top Q 个结果。Q 越大，我们需要保存的结果越多，但准确度也越高。
  * **输出：**
      * 动态搜索的准确度 / 端到端搜索的准确度。

-----

这份翻译和整理是否符合您的需求？如果您对其中的任何部分有疑问，或者想了解如何将这套工作流应用到其他类型的数据集上，请告诉我。