import os
import re
from torch.utils.data import Dataset
from PIL import Image
import torch
# 注意api下有原始的imagenet.py，这里是自己写的一个版本，主要是为了分离中英文类别名称

class ImageNetDataset(Dataset):
    def __init__(self, datadir, split="val", device="cuda:0", transform=None, annotation_file=None):
        self.datadir = datadir
        self.split = split
        self.transform = transform
        self.device = device

        # 加载ImageNet类别标签
        (self.text_list_en, self.text_list_zh,
         self.wnid_to_idx, self.idx_to_wnid,
         self.idx_to_class_en, self.idx_to_class_zh) = self._load_imagenet_classes(annotation_file)

        # 加载样本
        self.samples = []
        if split == "val":
            self._load_val_samples()
        elif split == "train":
            self._load_train_samples()

    def _load_imagenet_classes(self, annotation_file=None):
        """从annotation文件加载ImageNet类别信息，分离中英文"""
        if annotation_file is None:
            annotation_file = os.path.join(self.datadir, "annotation_LIST.txt")

        if not os.path.exists(annotation_file):
            print(f"警告: 未找到annotation文件 {annotation_file}，使用默认类别名")
            default_classes = [f"class_{i}" for i in range(1000)]
            return default_classes, default_classes, {}, {}, {}, {}

        text_list_en = []  # 英文类别名称列表
        text_list_zh = []  # 中文类别名称列表
        wnid_to_idx = {}
        idx_to_wnid = {}
        idx_to_class_en = {}  # 索引到英文类别名
        idx_to_class_zh = {}  # 索引到中文类别名

        with open(annotation_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # 解析每行: "0 n01440764 鱼, tench, Tinca tinca"
                parts = line.split(' ', 2)
                if len(parts) < 3:
                    continue

                idx = int(parts[0])
                wnid = parts[1]
                class_names = parts[2]

                # 分离中英文名称
                zh_name, en_names = self._separate_chinese_english(class_names)

                # 存储映射关系
                wnid_to_idx[wnid] = idx
                idx_to_wnid[idx] = wnid
                idx_class_en = en_names  # 英文名称（可能包含多个别名）
                idx_class_zh = zh_name  # 中文名称

                idx_to_class_en[idx] = idx_class_en
                idx_to_class_zh[idx] = idx_class_zh

                # 英文列表：使用第一个英文名称作为主要名称
                primary_en_name = en_names.split(', ')[0] if ', ' in en_names else en_names
                text_list_en.append(primary_en_name)
                text_list_zh.append(zh_name)

        print(f"成功加载 {len(text_list_en)} 个类别")
        print(f"示例 - 中文: {text_list_zh[0]}, 英文: {text_list_en[0]}")

        return text_list_en, text_list_zh, wnid_to_idx, idx_to_wnid, idx_to_class_en, idx_to_class_zh

    def _separate_chinese_english(self, class_names):
        """分离中英文类别名称"""
        # 使用正则表达式分离中文和英文
        # 中文通常在前，后面跟着英文（可能包含逗号分隔的多个英文名）
        zh_pattern = r'^([\u4e00-\u9fff]+)'
        match = re.match(zh_pattern, class_names)

        if match:
            zh_name = match.group(1)
            # 剩余部分是英文名称（去除开头的逗号和空格）
            en_part = class_names[len(zh_name):].strip()
            if en_part.startswith(','):
                en_part = en_part[1:].strip()
            en_names = en_part if en_part else zh_name  # 如果没有英文名，使用中文名
        else:
            # 如果没有中文，全部作为英文处理
            zh_name = class_names.split(',')[0]  # 使用第一个英文名作为中文显示
            en_names = class_names

        return zh_name, en_names

    def _load_val_samples(self):
        """加载验证集样本"""
        val_dir = os.path.join(self.datadir, 'val')

        if not os.path.exists(val_dir):
            print(f"错误: 验证集目录不存在 {val_dir}")
            return

        # 尝试不同的标签文件格式
        possible_label_files = [
            os.path.join(self.datadir, 'val.txt'),
            os.path.join(self.datadir, 'ILSVRC2012_validation_ground_truth.txt'),
            os.path.join(val_dir, 'val.txt')
        ]

        labels_file = None
        for file_path in possible_label_files:
            if os.path.exists(file_path):
                labels_file = file_path
                break

        if labels_file and os.path.exists(labels_file):
            # 方式1: 从标签文件加载
            with open(labels_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    if ' ' in line:
                        img_path, label = line.split(' ', 1)
                        full_path = os.path.join(val_dir, img_path)
                    else:
                        label = int(line)
                        img_name = f"ILSVRC2012_val_{int(label):08d}.JPEG"
                        full_path = os.path.join(val_dir, img_name)

                    if os.path.exists(full_path):
                        self.samples.append((full_path, int(label)))
        else:
            # 方式2: 直接从val目录扫描
            self._scan_image_directory(val_dir)

        print(f"加载 {len(self.samples)} 个验证集样本")

    def _load_train_samples(self):
        """加载训练集样本"""
        train_dir = os.path.join(self.datadir, 'train')

        if not os.path.exists(train_dir):
            print(f"错误: 训练集目录不存在 {train_dir}")
            return

        self._scan_image_directory(train_dir)
        print(f"加载 {len(self.samples)} 个训练集样本")

    def _scan_image_directory(self, directory):
        """扫描图像目录，支持按wnid分类或扁平结构"""
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)

            if os.path.isdir(item_path):
                # 如果是目录，假设是按wnid组织的
                label = self.wnid_to_idx.get(item, -1)
                if label != -1:
                    for img_file in os.listdir(item_path):
                        if self._is_image_file(img_file):
                            full_path = os.path.join(item_path, img_file)
                            self.samples.append((full_path, label))
            else:
                # 如果是文件，尝试从文件名推断标签
                if self._is_image_file(item):
                    label = self._infer_label_from_filename(item)
                    if label is not None:
                        self.samples.append((item_path, label))

    def _is_image_file(self, filename):
        """检查是否为图像文件"""
        return filename.lower().endswith(('.jpeg', '.jpg', '.png', '.bmp'))

    def _infer_label_from_filename(self, filename):
        """从文件名推断标签（针对标准ImageNet验证集）"""
        try:
            if filename.startswith('ILSVRC2012_val_'):
                # 标准ImageNet验证集文件名格式: ILSVRC2012_val_00000001.JPEG
                label_str = filename.split('_')[2].split('.')[0]
                return int(label_str) - 1  # 标签通常从1开始，需要减1
        except (ValueError, IndexError):
            pass
        return None

    def get_class_name_en(self, idx):
        """根据索引获取英文类别名称"""
        return self.idx_to_class_en.get(idx, f"class_{idx}")

    def get_class_name_zh(self, idx):
        """根据索引获取中文类别名称"""
        return self.idx_to_class_zh.get(idx, f"类别_{idx}")

    def get_primary_class_name_en(self, idx):
        """获取主要的英文类别名称（第一个英文名）"""
        full_name = self.get_class_name_en(idx)
        return full_name.split(',')[0].strip()

    def get_all_class_names_en(self, idx):
        """获取所有的英文类别名称（列表形式）"""
        full_name = self.get_class_name_en(idx)
        return [name.strip() for name in full_name.split(',')]

    def get_wnid(self, idx):
        """根据索引获取WNID"""
        return self.idx_to_wnid.get(idx, f"unknown_{idx}")

    def get_text_list_en(self):
        """返回英文类别文本列表（用于英文检索）"""
        return self.text_list_en

    def get_text_list_zh(self):
        """返回中文类别文本列表"""
        return self.text_list_zh

    def get_detailed_class_info(self, idx):
        """获取详细的类别信息"""
        return {
            'index': idx,
            'wnid': self.get_wnid(idx),
            'chinese': self.get_class_name_zh(idx),
            'english': self.get_class_name_en(idx),
            'primary_english': self.get_primary_class_name_en(idx),
            'all_english_names': self.get_all_class_names_en(idx)
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.long)

        except Exception as e:
            print(f"加载图像失败: {img_path}, 错误: {e}")
            blank_image = torch.zeros(3, 224, 224) if self.transform else Image.new('RGB', (224, 224))
            return blank_image, torch.tensor(label, dtype=torch.long)


# 使用示例
if __name__ == "__main__":
    import torchvision.transforms as transforms

    # 数据预处理
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集
    datadir = "../.datasets/imagenet"
    annotation_file = "../.datasets/imagenet/annotation_LIST.txt"

    dataset = ImageNetDataset(
        datadir=datadir,
        split="val",
        transform=data_transform,
        annotation_file=annotation_file
    )

    print(f"数据集大小: {len(dataset)}")
    print(f"类别数量: {len(dataset.text_list_en)}")

    # 测试类别信息获取
    if len(dataset.text_list_en) > 0:
        for i in range(3):  # 显示前3个类别的信息
            info = dataset.get_detailed_class_info(i)
            print(f"类别 {i}:")
            print(f"  WNID: {info['wnid']}")
            print(f"  中文: {info['chinese']}")
            print(f"  英文: {info['english']}")
            print(f"  主要英文: {info['primary_english']}")
            print(f"  所有英文名: {info['all_english_names']}")
            print()

    # 获取用于检索的英文文本列表
    retrieval_texts = dataset.get_text_list_en()
    print(f"用于检索的英文类别示例: {retrieval_texts[:5]}")