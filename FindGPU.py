import torch

# 检查是否有可用的 GPU
if torch.cuda.is_available():
    # 获取 GPU 数量
    gpu_count = torch.cuda.device_count()
    print(f"GPU Count: {gpu_count}")

    # 获取每个 GPU 的编号和名称
    for i in range(gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPU available.")
