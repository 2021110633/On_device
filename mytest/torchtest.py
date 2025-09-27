import torch


def test_pytorch_installation():
    print("=== PyTorch安装测试 ===")

    # 测试基本信息
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")

    # 测试CPU功能
    print("\n=== CPU测试 ===")
    cpu_tensor = torch.ones(3, 3)
    print(f"CPU张量:\n{cpu_tensor}")
    print(f"张量形状: {cpu_tensor.shape}")

    # 测试GPU功能（如果可用）
    if torch.cuda.is_available():
        print("\n=== GPU测试 ===")
        gpu_tensor = cpu_tensor.cuda()
        print(f"GPU张量:\n{gpu_tensor}")
        print(f"设备: {gpu_tensor.device}")

        # 测试GPU运算
        result = gpu_tensor * 2
        print(f"GPU运算结果:\n{result}")

    # 测试自动求导
    print("\n=== 自动求导测试 ===")
    x = torch.tensor(2.0, requires_grad=True)
    y = x ** 2
    y.backward()
    print(f"x=2时, dy/dx = {x.grad}")

    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    test_pytorch_installation()