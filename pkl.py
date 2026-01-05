import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import numpy as np
import pickle
from pathlib import Path
import os
import time


def convert_yolo_to_pkl(model_path, test_dir, output_pkl_path, conf_threshold=0.001):
    """
    将YOLO11-OBB推理结果转换为比赛要求的pkl格式

    Args:
        model_path: YOLO模型路径
        test_dir: 测试图片目录
        output_pkl_path: 输出pkl文件路径
        conf_threshold: 置信度阈值
    """
    # 开始计时
    start_time = time.time()
    
    # 加载模型
    print("正在加载模型...")
    model_load_start = time.time()
    model = YOLO(model_path)
    model_load_time = time.time() - model_load_start
    print(f"模型加载耗时: {model_load_time:.2f} 秒")

    # 定义类别映射
    class_mapping = {
        0: 'ship',
        1: 'aircraft',
        2: 'car',
        3: 'tank',
        4: 'bridge',
        5: 'harbor'
    }

    # 获取所有测试图片
    test_images = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        test_images.extend(list(Path(test_dir).glob(ext)))
    test_images = sorted(test_images)

    print(f"找到 {len(test_images)} 张测试图片")

    # 存储所有结果
    all_results = []
    
    # 推理计时
    inference_start = time.time()
    total_inference_time = 0

    # 处理每张图片
    for idx, img_path in enumerate(test_images, 1):
        print(f"处理图片 [{idx}/{len(test_images)}]: {img_path.name}", end=' ')
        
        # 单张图片推理计时
        img_start = time.time()

        # YOLO推理 - 添加了新的参数
        results = model(
            str(img_path),
            conf=conf_threshold,  # 使用传入的置信度阈值（默认0.001）
            iou=0.7,              # IoU阈值
            max_det=100,          # 最大检测数量
            imgsz=1024,
            verbose=False         # 关闭详细输出
        )
        result = results[0]  # 获取第一个结果
        
        img_time = time.time() - img_start
        total_inference_time += img_time

        # 初始化当前图片的检测结果
        image_result = {
            'image': img_path.name,
            'poly': np.empty((0, 8)),  # 空的numpy数组
            'scores': [],
            'labels': []
        }

        # 检查是否有检测结果
        if result.obb is not None and len(result.obb) > 0:
            # 获取检测数据
            xyxyxyxy = result.obb.xyxyxyxy.cpu().numpy()  # 形状: (N, 4, 2)
            confs = result.obb.conf.cpu().numpy()  # 形状: (N,)
            classes = result.obb.cls.cpu().numpy()  # 形状: (N,)

            # 转换数据格式
            polygons = []
            scores = []
            labels = []

            for i in range(len(result.obb)):
                # 获取4个顶点坐标并展平为8个值
                poly_points = xyxyxyxy[i].flatten()  # [x1,y1,x2,y2,x3,y3,x4,y4]

                # 获取置信度
                conf_score = float(confs[i])

                # 获取类别名称
                class_idx = int(classes[i])
                if class_idx in class_mapping:
                    class_name = class_mapping[class_idx]

                    polygons.append(poly_points)
                    scores.append(conf_score)
                    labels.append(class_name)
                else:
                    print(f"警告: 类别索引 {class_idx} 不在映射中，跳过")

            # 转换为numpy数组
            if len(polygons) > 0:
                image_result['poly'] = np.array(polygons)  # 形状: (n, 8)
                image_result['scores'] = scores
                image_result['labels'] = labels

        all_results.append(image_result)
        print(f"- 检测到 {len(image_result['scores'])} 个目标 (耗时: {img_time:.3f}秒)")

    inference_end = time.time()
    total_inference_time = inference_end - inference_start

    # 保存pkl文件计时
    save_start = time.time()
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(all_results, f)
    save_time = time.time() - save_start

    # 总时间
    total_time = time.time() - start_time

    # 打印时间统计
    print("\n" + "="*60)
    print("时间统计:")
    print(f"  模型加载时间: {model_load_time:.2f} 秒")
    print(f"  推理总时间: {total_inference_time:.2f} 秒")
    print(f"  平均每张图片: {total_inference_time/len(test_images):.3f} 秒")
    print(f"  保存pkl时间: {save_time:.2f} 秒")
    print(f"  总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
    print("="*60)
    print(f"\n结果已保存到: {output_pkl_path}")
    
    return all_results


def verify_pkl_format(pkl_path):
    """验证pkl文件格式是否正确"""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    print(f"\n验证pkl文件: {pkl_path}")
    print(f"总图片数: {len(data)}")

    # 检查前几个条目
    for i, entry in enumerate(data[:3]):
        print(f"\n图片 {i + 1}: {entry['image']}")
        print(f"  检测数量: {len(entry['poly'])}")
        print(f"  poly形状: {entry['poly'].shape}")
        if entry['scores']:
            print(f"  置信度样例: {entry['scores'][:3]}")
        if entry['labels']:
            print(f"  类别样例: {entry['labels'][:3]}")


def check_model_classes(model_path):
    """检查模型的类别信息"""
    model = YOLO(model_path)
    print("模型类别信息:")
    for idx, name in model.names.items():
        print(f"  索引 {idx}: {name}")
    return model.names


if __name__ == "__main__":
    # 设置路径
    model_path = "runs/train/exp3/weights/best.pt"  # 模型路径
    #test_dir = r"dataset/images/test_A"  # 初赛测试图片目录
    test_dir = r"dataset/test_B/images"  # 复赛测试图片目录
    output_pkl = "result.pkl"  # 输出pkl文件名

    # 先检查模型类别
    print("检查模型类别:")
    model_classes = check_model_classes(model_path)

    # 转换结果
    results = convert_yolo_to_pkl(
        model_path=model_path,
        test_dir=test_dir,
        output_pkl_path=output_pkl,
        conf_threshold=0.001
    )

    # 验证输出格式
    verify_pkl_format(output_pkl)