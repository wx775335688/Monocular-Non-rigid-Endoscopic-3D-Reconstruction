import json
from collections import OrderedDict

MASK_JSON = "/home/miccai/wx/data/sam-output/two/masks.json"  # 改成你的路径

def rgb_to_chinese(rgb):
    r, g, b = rgb
    if r > 200 and g < 80 and b < 80:
        return "红色"
    if g > 200 and r < 80 and b < 80:
        return "绿色"
    if b > 200 and r < 80 and g < 80:
        return "蓝色"
    if r > 200 and g > 200 and b < 80:
        return "黄色"
    if r > 200 and b > 200 and g < 80:
        return "品红/紫红"
    if g > 200 and b > 200 and r < 80:
        return "青色"
    if r > 200 and g > 120 and b < 80:
        return "橙色"
    if r > 120 and b > 200 and g < 120:
        return "紫色/蓝紫"
    if g > 120 and b > 200 and r < 120:
        return "青蓝/湖蓝"
    if r > 120 and g > 200 and b < 120:
        return "黄绿"
    return "其他混合色"

def load_name_color_map(path: str):
    with open(path, "r") as f:
        data = json.load(f)

    name2color = OrderedDict()
    for frame_key, frame_data in data.items():
        for cls in frame_data.get("classes", []):
            name = cls.get("name")
            color = cls.get("color")
            if name is None or color is None:
                continue
            if name not in name2color:
                name2color[name] = color
    return name2color

if __name__ == "__main__":
    name2color = load_name_color_map(MASK_JSON)
    print("器官与颜色对应（中文描述）：")
    for name, color in name2color.items():
        print(f"{name}: {color} -> {rgb_to_chinese(color)}")