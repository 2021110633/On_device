
-----

# 非官方 ImageBind LoRA 微调

这是一个非官方的 **ImageBind Trainer** 实现，支持 **LoRA 微调**。要将此仓库适配到您自己的数据集，请查看 `train.py` 并将其中的 `dreambooth` 替换为您的数据集名称。

请确保递归克隆此仓库以包含子模块：

```
git clone --recurse-submodules -j8 https://github.com/fabawi/ImageBind-LoRA.git
```

有关安装，请遵循 [原始使用说明](https://github.com/facebookresearch/ImageBind)。在使用不带 `--headless` 参数的 `train.py` 脚本时，请安装 `matplotlib`。

**警告：** 如果您收到此错误 -\> `'FastAPI' object has no attribute 'debug'`，请将 `fastapi` 升级到最新版本：

```
pip install --upgrade fastapi
```

-----

## 搜索

在 `search.py` 中，您可以找到一个如何使用模型搜索目标标签图像的示例。

  * 要尝试 **LoRA 微调模型**，请将 `lora` 设置为 `True`，并在脚本中设置微调模型的路径 `lora_dir` 和 `LoRA.apply_lora_modality_trunks()` 中的参数。
  * 要尝试 **原始 ImageBind 模型**，请将 `lora` 设置为 `False`。

**示例说明：**
**Imagenet** 数据集包含 1000 个类别，搜索对应 "stingray", "cock", "hen" 这三个词的图像，并获取我们搜索到的图像的精确率和召回率。

-----

## 微调

修改 `train.py` 以适应 ImageNet 数据集的训练，修改后的代码存储在 `train_imagenet.py` 中。

以下是关于 `train.py` 的信息。要训练模型，请运行：

```
python train.py --batch_size 12 --max_epochs 500 \
        --lora --lora_modality_names vision text \
        --self_contrast --datasets dreambooth
```

您可以通过设置 `--loggers` 参数来启用 `comet`、`wandb` 或 `tensorboard` 进行日志记录。请确保事先安装相应的日志包和必要的环境变量。

要指定对哪些层或模态应用 LoRA，请使用 `--lora_layer_idxs` 和 `--lora_modality_names` 参数。要覆盖某个特定模态的层数，可以专门指定该模态，例如，添加以下参数来为视觉主干的前 6 层指定 LoRA：

```
--lora_layer_idxs_vision 1 2 3 4 5 6
```

要在 GPU 上训练（目前支持单个 GPU，多 GPU 训练即将添加），请设置 `--device` 参数：

```
--device cuda:0
```

在 `example.py` 中使用的 LoRA 模型（检查点位于 `.checkpoints/lora/550_epochs/`，后缀为 `_dreambooth_last.safetensors`）在 **12 GB VRAM 的 3080Ti** 上训练了约 2 小时，消耗了 5.66 GB VRAM 和约 4 GB RAM。该模型在不到 30 分钟内收敛到了相似的状态。

**信息：**

  * 8.0 M 可训练参数
  * 1.2 B 不可训练参数
  * 1.2 B 总参数
  * 4,815.707 总模型估计参数大小 (MB)

我们设置的训练参数如下：

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
要执行线性探测（仅优化每个模态头部中的最后一层），请保留所有参数，将 `--lora` 替换为 `--linear_probing`（两者不能在同一次运行中同时设置）。在接下来的训练会话中运行 `--lora` 时，检查点会自动加载和保存，前提是 `--lora_checkpoint_dir` 保持不变。

-----

## 推理

在 `test_imagenet.py` 中，您可以找到一个在 ImageNet 数据集上使用模型进行推理的示例。

  * 要尝试 **LoRA 微调模型**，请将 `lora` 设置为 `True`，并在脚本中设置微调模型的路径 `lora_dir` 和 `LoRA.apply_lora_modality_trunks()` 中的参数。
  * 要尝试 **原始 ImageBind 模型**，请将 `lora` 设置为 `False`。

您还可以使用 `imagebind_model.imagebind_huge()` 设置每个模态的主干块。

-----

## ImageBind：一个嵌入空间，绑定所有模态

  * **作者：** Rohit Girdhar\*, Alaaeldin El-Nouby\*, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, Ishan Misra\*
  * **机构：** FAIR, Meta AI
  * **论文：** 即将发表在 CVPR 2023（亮点论文）

[论文](https://arxiv.org/abs/2305.10702) [博客](https://www.google.com/search?q=https://ai.meta.com/blog/imagebind-six-modalities-one-embedding-space/) [演示](https://imagebind.metademolab.com) [补充视频](https://www.google.com/search?q=https://www.youtube.com/watch%3Fv%3DkYJv9cE7vD4) [BibTex]

这是 **ImageBind** 的 **PyTorch** 实现和预训练模型。有关详细信息，请参阅论文：《ImageBind: One Embedding Space To Bind Them All》。

ImageBind 学习一个跨越六种不同模态（图像、文本、音频、深度、热力图和 IMU 数据）的联合嵌入空间。它支持“开箱即用”的创新突现式应用，包括跨模态检索、用算术组合模态、跨模态检测和生成。

-----

## ImageBind 模型

突现式的零样本分类性能。
| Model | IN1k | K400 | NYU-DESC | LLVI | PEgo4D |
| :--- | :---: | :---: | :---: | :---: | :---: |
| imagebind\_huge | 77.7 | 50.0 | 54.0 | 66.9 | 63.4 |

-----

## 使用

1.  安装 **PyTorch 1.13+** 和其他第三方依赖。
    ```
    conda create --name imagebind python=3.8 -y
    conda activate imagebind
    pip install -r requirements.txt
    ```
2.  对于 Windows 用户，您可能需要安装 `soundfile` 来读写音频文件。（感谢 @congyue1977）
    ```
    pip install soundfile
    ```

### 跨模态特征提取和比较（例如：图像、文本和音频）

```python
import data
import torch
from models import imagebind_model
from models.imagebind_model import ModalityType

text_list=["A dog.", "A car", "A bird"]
image_paths=[".assets/dog_image.jpg", ".assets/car_image.jpg", ".assets/bird_image.jpg"]
audio_paths=[".assets/dog_audio.wav", ".assets/car_audio.wav", ".assets/bird_audio.wav"]

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 实例化模型
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

# 加载数据
inputs = {
    ModalityType.TEXT: data.load_and_transform_text(text_list, device),
    ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
    ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device),
}

with torch.no_grad():
    embeddings = model(inputs)

print(
    "Vision x Text: ",
    torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1),
)
print(
    "Audio x Text: ",
    torch.softmax(embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T, dim=-1),
)
print(
    "Vision x Audio: ",
    torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.AUDIO].T, dim=-1),
)
# 预期输出:
## Vision x Text:
# tensor([[9.9761e-01, 2.3694e-03, 1.8612e-05],
#         [3.3836e-05, 9.9994e-01, 2.4118e-05],
#         [4.7997e-05, 1.3496e-02, 9.8646e-01]])
## Audio x Text:
# tensor([[1., 0., 0.],
#         [0., 1., 0.],
#         [0., 0., 1.]])
## Vision x Audio:
# tensor([[0.8070, 0.1088, 0.0842],
#         [0.1036, 0.7884, 0.1079],
#         [0.0018, 0.0022, 0.9960]])
```

-----

## 模型卡

请参阅 [模型卡](https://www.google.com/search?q=https://github.com/facebookresearch/ImageBind/blob/main/MODEL_CARD.md) 了解详细信息。

-----

## 许可证

ImageBind 的代码和模型权重根据 **CC-BY-NC 4.0** 许可证发布。请参阅 [LICENSE](https://www.google.com/search?q=https://github.com/facebookresearch/ImageBind/blob/main/LICENSE) 文件获取更多详细信息。

-----

## 贡献

请参阅 [contributing](https://github.com/facebookresearch/ImageBind/blob/main/CONTRIBUTING.md) 和 [行为准则](https://github.com/facebookresearch/ImageBind/blob/main/CODE_OF_CONDUCT.md)。

-----

## 引用 ImageBind

如果您发现此仓库有用，请考虑给一个 star :star: 并引用：

```
@inproceedings{girdhar2023imagebind,
  title={ImageBind: One Embedding Space To Bind Them All},
  author={Girdhar, Rohit and El-Nouby, Alaaeldin and Liu, Zhuangand Singh, Mannat and Alwala, Kalyan Vasudev and Joulin, Armand and Misra, Ishan},
  booktitle={CVPR},
  year={2023}
}
```

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

其他参数（大多数可以在 “parser” 函数中更改）：

  * `lora_dir`:
    ```
    parser.add_argument("--lora_checkpoint_dir", type=str, default="./.checkpoints/lora/lume_lora-666!!!")
    ```
  * `device`:
    ```
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training and validation")
    ```
  * ...
    ```
    trainer = Trainer(accelerator="gpu" if "cuda" in device_name else "cpu", devices=[1,2,3], deterministic=True)
    ```

这份翻译详细地涵盖了您提供的所有内容，并根据其原始结构进行了组织。

如果您需要将这份 README 翻译成其他语言，或者想了解如何开始使用这个项目，我可以为您提供更多帮助。