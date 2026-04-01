import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import math
import os

# ==========================================
# 1. 配置与常量 (必须与训练时完全一致)
# ==========================================
IMAGE_SIZE = 224
MINUTES_PER_CYCLE = 720

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ==========================================
# 2. 模型定义 (必须与训练时完全一致)
# ==========================================
import torchvision.models as models

class ClockResNet(nn.Module):
    def __init__(self):
        super(ClockResNet, self).__init__()
        # 即使是推理，也要实例化同样的结构
        self.resnet = models.resnet18(weights=None) # 推理时不需下载预训练权重
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 4)

    def forward(self, x):
        out = self.resnet(x)
        # 核心：推理时也必须进行向量归一化
        out_h = F.normalize(out[:, 0:2], p=2, dim=1)
        out_m = F.normalize(out[:, 2:4], p=2, dim=1)
        return torch.cat([out_h, out_m], dim=1)

# ==========================================
# 3. 核心推理函数
# ==========================================
def predict_single_image(image_path, model_path="clock_model.pth"):
    device = get_device()
    print(f"Using device: {device}")

    # A. 加载模型
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return

    model = ClockResNet().to(device)
    # 加载保存的权重 (map_location 确保在没有GPU时也能加载CUDA模型)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # 必须设置为评估模式
    print("Model loaded successfully.")

    # B. 图像预处理
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return

    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    # 定义与训练时完全一样的 transform
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # 对图片进行处理，并增加一个 batch 维度 [1, 3, 224, 224]
    input_tensor = transform(image).unsqueeze(0).to(device)

    # C. 运行推理
    with torch.no_grad(): # 推理时不需要计算梯度
        outputs = model(input_tensor)

    # D. 高精度解码 (从向量到时间)
    # 取出预测的向量 (注意：此时 output 是 [1, 4]，我们需要取出里面的标量)
    pred_sin_h, pred_cos_h = outputs[0, 0], outputs[0, 1]
    pred_sin_m, pred_cos_m = outputs[0, 2], outputs[0, 3]

    # 计算角度 (0 到 2pi)
    pred_angle_h = torch.remainder(torch.atan2(pred_sin_h, pred_cos_h), 2 * math.pi)
    pred_angle_m = torch.remainder(torch.atan2(pred_sin_m, pred_cos_m), 2 * math.pi)

    # 双指针结合逻辑
    pred_min_part = (pred_angle_m / (2 * math.pi)) * 60.0
    pred_total_from_h = (pred_angle_h / (2 * math.pi)) * MINUTES_PER_CYCLE
    
    # 用时针推算小时(0-11)，用分针确定分钟
    pred_hour = torch.round((pred_total_from_h - pred_min_part) / 60.0) % 12
    pred_minutes = torch.round(pred_hour * 60 + pred_min_part)

    # 限制范围在 0~719 之间，防止 round 导致的越界
    pred_minutes_final = torch.clamp(pred_minutes, 0, MINUTES_PER_CYCLE - 1)

    # 格式化输出
    final_minutes = int(pred_minutes_final.item())
    display_hour = final_minutes // 60
    display_minute = final_minutes % 60
    
    # 模拟传统时钟，将 0 点显示为 12 点
    print_hour = 12 if display_hour == 0 else display_hour

    print("-" * 30)
    print(f"测试图片: {image_path}")
    print(f"预测时间: {print_hour:02}:{display_minute:02}")
    print("-" * 30)

# ==========================================
# 4. 程序入口：在这里修改你的图片路径
# ==========================================
if __name__ == "__main__":
    # 📢 修改这里！ 
    # 指向你想测试的单张图片路径 (例如: "my_clock.jpg" 或 "data/test/clock_001.png")
    your_image_path = "test_image02.png" 
    
    predict_single_image(your_image_path)
