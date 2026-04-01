import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, random_split

from PIL import Image
import torchvision.transforms as transforms

import json
import os


IMAGE_SIZE = 300
NUM_HOUR_CLASSES = 12


#################################
# 时间转换函数
#################################

def time_to_hour_class(time_str):

    hour, _ = map(int, time_str.split(":"))

    # 12点映射到类别0，其余1-11点映射到1-11
    return hour % 12


#################################
# Dataset
#################################

class ClockHourDataset(Dataset):

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

        label = time_to_hour_class(item["time"])

        return image, torch.tensor(label, dtype=torch.long)


#################################
# CNN模型（小时分类）
#################################

class ClockHourCNN(nn.Module):

    def __init__(self):

        super(ClockHourCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))

        self.fc1 = nn.Linear(128 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, NUM_HOUR_CLASSES)

    def forward(self, x):

        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.adaptive_pool(x)

        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x


#################################
# 主训练函数
#################################

def train():

    dataset = ClockHourDataset(
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

    model = ClockHourCNN()

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    epochs = 20

    for epoch in range(epochs):

        model.train()
        total_loss = 0.0

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
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    test_accuracy = 0.0 if total == 0 else correct / total

    print(f"测试集样本数: {total}")
    print(f"小时分类准确率: {test_accuracy:.2%}")

    torch.save(
        model.state_dict(),
        "clock_hour_model.pth"
    )

    print("小时分类模型训练完成并保存！")


#################################
# 程序入口
#################################

if __name__ == "__main__":

    train()
