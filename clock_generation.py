from PIL import Image, ImageDraw, ImageFont
import math
import random
import json
import os


# 五种背景颜色
BG_COLORS = [
    "white",
    "lightyellow",
    "lightblue",
    "lightgreen",
    "lavender"
]

# 统一输出分辨率（宽高相同）
IMAGE_SIZE = 300


def draw_clock(hour, minute, filename):

    size = IMAGE_SIZE
    center = size // 2
    radius = int(size * 0.39)

    # 随机背景颜色
    bg_color = random.choice(BG_COLORS)

    img = Image.new("RGB", (size, size), bg_color)
    draw = ImageDraw.Draw(img)

    # 外圆
    draw.ellipse(
        (center-radius, center-radius,
         center+radius, center+radius),
        outline="black",
        width=3
    )

    # 画刻度线
    for i in range(60):

        angle = math.radians(i * 6 - 90)

        inner = radius - 10
        outer = radius

        if i % 5 == 0:
            inner = radius - 20  # 大刻度

        x1 = center + inner * math.cos(angle)
        y1 = center + inner * math.sin(angle)

        x2 = center + outer * math.cos(angle)
        y2 = center + outer * math.sin(angle)

        draw.line((x1, y1, x2, y2), fill="black", width=2)

    # 画数字 1–12
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()

    for num in range(1, 13):

        angle = math.radians(num * 30 - 90)

        text_radius = radius - 35

        x = center + text_radius * math.cos(angle)
        y = center + text_radius * math.sin(angle)

        draw.text(
            (x-5, y-5),
            str(num),
            fill="black",
            font=font
        )

    # 指针角度
    hour_angle = (hour % 12) * 30 + minute * 0.5
    minute_angle = minute * 6

    hour_rad = math.radians(hour_angle - 90)
    minute_rad = math.radians(minute_angle - 90)

    hour_length = radius * 0.5
    minute_length = radius * 0.8

    hour_x = center + hour_length * math.cos(hour_rad)
    hour_y = center + hour_length * math.sin(hour_rad)

    minute_x = center + minute_length * math.cos(minute_rad)
    minute_y = center + minute_length * math.sin(minute_rad)

    # 时针
    draw.line(
        (center, center, hour_x, hour_y),
        fill="black",
        width=6
    )

    # 分针
    draw.line(
        (center, center, minute_x, minute_y),
        fill="black",
        width=3
    )

    # # 随机旋转
    # rotate_angle = random.randint(0, 359)
    # img = img.rotate(rotate_angle, resample=Image.BICUBIC, expand=False)
    img = img.rotate(0, resample=Image.BICUBIC, expand=False)
    
    # 兜底保证输出尺寸统一
    if img.size != (IMAGE_SIZE, IMAGE_SIZE):
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)

    img.save(filename)


def generate_dataset(num_images=1000):

    os.makedirs("clocks_test", exist_ok=True)

    dataset = []

    for i in range(num_images):

        hour = random.randint(0, 23)
        minute = random.randint(0, 59)

        hour_12 = hour % 12
        if hour_12 == 0:
            hour_12 = 12

        time_str = f"{hour_12:02d}:{minute:02d}"

        filename = f"clock_{i:05d}.png"
        filepath = os.path.join("clocks", filename)

        draw_clock(hour, minute, filepath)

        dataset.append({
            "image": filename,
            "time": time_str
        })

    with open("clocks_test/labels.json", "w") as f:
        json.dump(dataset, f, indent=4)

    print("数据集生成完成！")


generate_dataset(10)