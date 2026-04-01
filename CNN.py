import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import json
import os
import math

# 适配ResNet标准输入
IMAGE_SIZE = 224 
MINUTES_PER_CYCLE = 720

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

#################################
# 时间与角度转换函数
#################################

def time_to_minutes(time_str):
    hour, minute = map(int, time_str.split(":"))
    return (hour % 12) * 60 + minute

def circular_minute_error(pred_minutes, true_minutes):
    diff = torch.abs(pred_minutes - true_minutes)
    return torch.minimum(diff, MINUTES_PER_CYCLE - diff)

#################################
# Dataset (Sin/Cos 编码)
#################################

class ClockDataset(Dataset):
    def __init__(self, json_path, image_dir):
        with open(json_path) as f:
            self.data = json.load(f)
        self.image_dir = image_dir
        
        # 数据标准化，适配预训练模型
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.image_dir, item["image"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        minutes = time_to_minutes(item["time"])
        
        # 1. 计算时针角度 (映射到 0~2pi)
        theta_hour = (minutes / MINUTES_PER_CYCLE) * 2 * math.pi
        
        # 2. 计算分针角度 (映射到 0~2pi)
        minute_of_hour = minutes % 60
        theta_minute = (minute_of_hour / 60.0) * 2 * math.pi
        
        # 标签：[时针sin, 时针cos, 分针sin, 分针cos]
        label = torch.tensor([
            math.sin(theta_hour), math.cos(theta_hour),
            math.sin(theta_minute), math.cos(theta_minute)
        ], dtype=torch.float32)

        return image, label

#################################
# 深度模型 (ResNet18 + 向量归一化)
#################################

class ClockResNet(nn.Module):
    def __init__(self):
        super(ClockResNet, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 4)

    def forward(self, x):
        out = self.resnet(x)
        
        # 强制将预测的 (sin, cos) 向量拉伸到单位圆上，消除向量长度带来的误差
        out_h = F.normalize(out[:, 0:2], p=2, dim=1)
        out_m = F.normalize(out[:, 2:4], p=2, dim=1)
        
        return torch.cat([out_h, out_m], dim=1)

#################################
# 主训练与评估函数
#################################

def train():
    device = get_device()
    print(f"当前计算设备: {device}")

    use_pin_memory = device.type == "cuda"

    # 请确保路径和你的本地文件结构一致
    dataset = ClockDataset(
        json_path="clocks/labels.json",
        image_dir="clocks"
    )

    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size

    train_set, test_set = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=32,
        shuffle=True,
        pin_memory=use_pin_memory,
        num_workers=2
    )

    test_loader = DataLoader(
        test_set,
        batch_size=32,
        shuffle=False,
        pin_memory=use_pin_memory,
        num_workers=2
    )

    model = ClockResNet().to(device)
    criterion = nn.MSELoss()

    # 初始学习率稍微大一点
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    
    # 学习率调度器：每 30 轮，学习率减半，帮助模型在后期更好地收敛
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    epochs = 100

    print("开始训练...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=use_pin_memory)
            labels = labels.to(device, non_blocking=use_pin_memory)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    # ==========================
    # 测试与评估阶段 (双指针高精度推断)
    # ==========================
    model.eval()
    tolerance_minutes = 5
    strict_correct = 0
    tolerant_correct = 0
    abs_error_sum = 0.0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=use_pin_memory)
            labels = labels.to(device, non_blocking=use_pin_memory)

            outputs = model(images)

            # 1. 取出预测与真实的向量
            pred_sin_h, pred_cos_h = outputs[:, 0], outputs[:, 1]
            pred_sin_m, pred_cos_m = outputs[:, 2], outputs[:, 3]
            
            true_sin_h, true_cos_h = labels[:, 0], labels[:, 1]
            true_sin_m, true_cos_m = labels[:, 2], labels[:, 3]

            # 2. 计算角度 (映射到 0 到 2pi)
            pred_angle_h = torch.remainder(torch.atan2(pred_sin_h, pred_cos_h), 2 * math.pi)
            pred_angle_m = torch.remainder(torch.atan2(pred_sin_m, pred_cos_m), 2 * math.pi)
            
            true_angle_h = torch.remainder(torch.atan2(true_sin_h, true_cos_h), 2 * math.pi)
            true_angle_m = torch.remainder(torch.atan2(true_sin_m, true_cos_m), 2 * math.pi)

            # 3. 双指针结合计算精准时间
            # 预测时间
            pred_min_part = (pred_angle_m / (2 * math.pi)) * 60.0
            pred_total_from_h = (pred_angle_h / (2 * math.pi)) * MINUTES_PER_CYCLE
            # 用时针推算小时(0-11)，用分针确定分钟
            pred_hour = torch.round((pred_total_from_h - pred_min_part) / 60.0) % 12
            pred_minutes = torch.round(pred_hour * 60 + pred_min_part)

            # 真实时间
            true_min_part = (true_angle_m / (2 * math.pi)) * 60.0
            true_total_from_h = (true_angle_h / (2 * math.pi)) * MINUTES_PER_CYCLE
            true_hour = torch.round((true_total_from_h - true_min_part) / 60.0) % 12
            true_minutes = torch.round(true_hour * 60 + true_min_part)

            # 限制范围在 0~719 之间
            pred_minutes = torch.clamp(pred_minutes, 0, MINUTES_PER_CYCLE - 1)
            true_minutes = torch.clamp(true_minutes, 0, MINUTES_PER_CYCLE - 1)

            # 4. 计算误差
            minute_error = circular_minute_error(pred_minutes, true_minutes)

            strict_correct += (minute_error == 0).sum().item()
            tolerant_correct += (minute_error <= tolerance_minutes).sum().item()
            abs_error_sum += minute_error.sum().item()
            total += labels.size(0)

    strict_accuracy = 0.0 if total == 0 else strict_correct / total
    tolerant_accuracy = 0.0 if total == 0 else tolerant_correct / total
    mae_minutes = 0.0 if total == 0 else abs_error_sum / total

    print("-" * 40)
    print(f"测试集样本数: {total}")
    print(f"严格准确率(误差=0分钟): {strict_accuracy:.2%}")
    print(f"容差准确率(误差<={tolerance_minutes}分钟): {tolerant_accuracy:.2%}")
    print(f"测试集平均绝对误差(MAE): {mae_minutes:.2f} 分钟")

    # 保存模型
    torch.save(model.state_dict(), "clock_model.pth")
    print("模型训练完成并保存！")

#################################
# 程序入口
#################################
if __name__ == "__main__":
    train()