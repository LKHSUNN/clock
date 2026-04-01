import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, random_split

from PIL import Image
import torchvision.transforms as transforms

import json
import os


IMAGE_SIZE = 300
MINUTES_PER_CYCLE = 720


#################################
# 时间转换函数
#################################

def time_to_minutes(time_str):

    hour, minute = map(int, time_str.split(":"))

    return (hour % 12) * 60 + minute


def circular_minute_error(pred_minutes, true_minutes):

    diff = torch.abs(pred_minutes - true_minutes)

    return torch.minimum(diff, MINUTES_PER_CYCLE - diff)


#################################
# Dataset
#################################

class ClockDataset(Dataset):

    def __init__(self, json_path, image_dir):

        with open(json_path) as f:
            self.data = json.load(f)

        self.image_dir = image_dir

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor()
        ])

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        item = self.data[idx]

        img_path = os.path.join(
            self.image_dir,
            item["image"]
        )

        image = Image.open(img_path).convert("RGB")

        image = self.transform(image)

        minutes = time_to_minutes(item["time"])

        label = minutes / MINUTES_PER_CYCLE

        return image, torch.tensor([label], dtype=torch.float32)


#################################
# CNN模型
#################################

class ClockCNN(nn.Module):

    def __init__(self):

        super(ClockCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))

        self.fc1 = nn.Linear(128 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):

        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.adaptive_pool(x)

        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))

        x = torch.sigmoid(self.fc2(x))

        return x


#################################
# 主训练函数
#################################

def train():

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
        shuffle=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=32,
        shuffle=False
    )

    model = ClockCNN()

    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    epochs = 20

    for epoch in range(epochs):

        total_loss = 0

        for images, labels in train_loader:

            outputs = model(images)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}"
        )

    model.eval()
    tolerance_minutes = 5
    strict_correct = 0
    tolerant_correct = 0
    abs_error_sum = 0.0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:

            outputs = model(images)

            pred_minutes = torch.round(outputs.squeeze(1) * MINUTES_PER_CYCLE)
            pred_minutes = torch.clamp(pred_minutes, 0, MINUTES_PER_CYCLE - 1)
            true_minutes = torch.round(labels.squeeze(1) * MINUTES_PER_CYCLE)

            minute_error = circular_minute_error(pred_minutes, true_minutes)

            strict_correct += (minute_error == 0).sum().item()
            tolerant_correct += (minute_error <= tolerance_minutes).sum().item()
            abs_error_sum += minute_error.sum().item()
            total += labels.size(0)

    strict_accuracy = 0.0 if total == 0 else strict_correct / total
    tolerant_accuracy = 0.0 if total == 0 else tolerant_correct / total
    mae_minutes = 0.0 if total == 0 else abs_error_sum / total

    print(f"测试集样本数: {total}")
    print(f"严格准确率(误差=0分钟): {strict_accuracy:.2%}")
    print(f"容差准确率(误差<={tolerance_minutes}分钟): {tolerant_accuracy:.2%}")
    print(f"测试集平均绝对误差(MAE): {mae_minutes:.2f} 分钟")

    # 保存模型
    torch.save(
        model.state_dict(),
        "clock_model.pth"
    )

    print("模型训练完成并保存！")


#################################
# 程序入口
#################################

if __name__ == "__main__":

    train()