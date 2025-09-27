import sys
import os

# Get the absolute path of the directory containing your project's root
# This line assumes your script is in a subdirectory (e.g., mytest)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
# datasets_path = os.path.join(project_root, "datasets")
# if datasets_path not in sys.path:
#     sys.path.append(datasets_path)
# print(f"-------------------Project root added to sys.path: {project_root}")
# print(f"-------------------Datasets path added to sys.path: {datasets_path}")

import logging
import torch
import data
import torchvision
import torchmetrics

from models import imagebind_model
from models.imagebind_model import ModalityType, load_module
from models import lora as LoRA


from metrics.search import Search
from api.imagenet import ImageNetDataset

from torchvision import transforms
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader


'''你的程序主要是在执行一个多模态（图像和文本）的检索任务，它使用 ImageBind 模型来评估在 ImageNet 验证集上的性能。

下面是程序的整体流程以及一些关键步骤的详细说明：

1. 初始化和配置
参数设置: 程序首先定义了一系列参数，比如是否使用 LoRA (一种微调技术)、是否进行线性探测、使用的设备（CPU 或 GPU），以及数据集和权重文件的路径。

模型加载: 程序加载了一个预训练的 imagebind_huge 模型。如果启用了 LoRA，它会进一步对模型的主干网络应用 LoRA 模块并加载预训练的 LoRA 权重。

评估模式: 模型被设置为评估模式 (model.eval()) 并移动到指定的设备上。这会禁用 dropout 等训练时的行为，以确保推理结果的一致性。

2. 数据准备
数据预处理: run_inference 函数定义了一系列图像预处理步骤，包括调整大小、中心裁剪和归一化。这些步骤都是为了让图像数据符合 ImageBind 模型的输入要求。

数据集和数据加载器: 程序创建了一个 ImageNetDataset 实例来加载 ImageNet 验证集，并用 DataLoader 来分批次处理数据。

内存密集型操作: 在这里，DataLoader 会加载 ImageNet 验证集的图像。更重要的是，在 for 循环内部，data.load_and_transform_text 函数会加载所有 1000 个 ImageNet 类别标签。这正是程序中最占用内存的地方。它会一次性为所有类别生成嵌入向量，当内存不足时就会引发你遇到的 RuntimeError。

3. 推理和评估
前向传播: 在循环中，程序将图像和所有文本标签作为输入传递给 model。模型会为这些多模态数据生成嵌入向量。

计算相似度: 程序计算了图像嵌入和所有文本嵌入之间的相似度矩阵。这个矩阵的大小是 [batch_size, 1000]，表示每个图像与 1000 个类别标签的相似度。

评估指标更新: 最后，test_search 实例使用这个相似度矩阵和图像的真实标签来更新精度和召回率等评估指标。'''

# 配置日志记录
logging.basicConfig(level=logging.INFO, force=True)

# 配置参数
lora = False # 是否使用LoRA微调
linear_probing = False # 是否使用线性探测
device = "cuda:1" if torch.cuda.is_available() else "cpu"  # 设备选择
print(f"Using device: {device}")
# device =  "cpu"  # 设备选择
load_head_post_proc_finetuned = True # 是否加载微调后的后处理器和头部
# datadir = "/data/yx/ImageBind/.datasets/imagenet" # 数据集路径
datadir = "../.datasets/imagenet"  # 数据集路径

lora_dir = '' 
# 检查参数冲突：LoRA和线性探测不能同时启用
assert not (linear_probing and lora), \
            "Linear probing is a subset of LoRA training procedure for ImageBind. " \
            "Cannot set both linear_probing=True and lora=True. "

# 设置LoRA因子（用于调整相似度计算）
if lora and not load_head_post_proc_finetuned:
    # Hack: adjust lora_factor to the `max batch size used during training / temperature` to compensate missing norm
    # 调整因子以补偿训练时缺失的归一化
    lora_factor = 12 / 0.07
else:
    # This assumes proper loading of all params but results in shift from original dist in case of LoRA
    # 正常情况下的因子
    lora_factor = 1


# Instantiate model
# 实例化预训练的ImageBind模型
model = imagebind_model.imagebind_huge(pretrained=True)
# 如果启用LoRA，应用LoRA到模型的主干网络
if lora:
    model.modality_trunks.update(
        LoRA.apply_lora_modality_trunks(model.modality_trunks, rank=4,
                                        layer_idxs={ModalityType.TEXT: [ 1, 2, 3, 4, 5, 6, 7, 8],
                                                    ModalityType.VISION: [1, 2, 3, 4, 5, 6, 7, 8]},
                                        modality_names=[ModalityType.TEXT, ModalityType.VISION]))

    # Load LoRA params if found
    # 加载预训练的LoRA参数
    LoRA.load_lora_modality_trunks(model.modality_trunks,
                                   checkpoint_dir=lora_dir)

    if load_head_post_proc_finetuned:
        # Load postprocessors & heads
        load_module(model.modality_postprocessors, module_name="postprocessors",
                    checkpoint_dir=lora_dir)
        load_module(model.modality_heads, module_name="heads",
                    checkpoint_dir=lora_dir)
# 如果启用线性探测，只加载分类头
elif linear_probing:
    # Load heads
    load_module(model.modality_heads, module_name="heads",
                checkpoint_dir=lora_dir)

model.eval()# 设置模型为评估模式
model.to(device)# 将模型移动到指定设备（GPU或CPU）


def run_inference():
    """运行推理函数：在ImageNet验证集上评估模型性能"""
     # 定义图像预处理流程
    data_transform = transforms.Compose(
        [
            transforms.Resize(
                224, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    
    test_ds = ImageNetDataset(datadir=datadir, split="val", device=device, transform=data_transform)
    test_dl = DataLoader(dataset=test_ds, batch_size=64, shuffle=False, drop_last=False,
        num_workers=4, pin_memory=True, persistent_workers=True)
    
    test_search = Search(num_classes=1000, device=device, target_classes=[test_ds.text_list.index('stingray'),
                                         test_ds.text_list.index('cock'),
                                         test_ds.text_list.index('hen')])


    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_dl):
            if torch.min(target).item() > test_ds.text_list.index('hen') or torch.max(target).item() < test_ds.text_list.index('stingray'):
                continue

            # feats_a = [model({class_a[0]: data_a_i}) for data_a_i in data_a]
            # feats_a_tensor = torch.cat([list(dict_.values())[0] for dict_ in feats_a], dim=0)
            # # class_b could be any modality
            # feats_b = [model({class_b[idx]: data_b_i}) for idx, data_b_i in enumerate(data_b)]
            # feats_b_tensor = torch.cat([list(dict_.values())[0] for dict_ in feats_b], dim=0)

            # match_value = feats_a_tensor @ feats_b_tensor.T

            x = x.to(device)
            target = target.to(device)
            inputs = {
                ModalityType.VISION: x,
                ModalityType.TEXT: data.load_and_transform_text(test_ds.text_list, device),
            }
            
            embeddings = model(inputs)
            match_value_1 = embeddings[ModalityType.VISION]@embeddings[ModalityType.TEXT].T * (lora_factor if lora else 1)

            result_1 = torch.softmax(match_value_1, dim=-1)
            test_search.update(result_1,target)
        
        test_search.compute()
    
    print(test_search.pricision)  # 打印精度
    print(test_search.recall)  # 打印召回率
        


if __name__ == "__main__":
    run_inference()

