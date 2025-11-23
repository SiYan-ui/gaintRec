import os
import cv2
import numpy as np
import argparse
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

def extract_skeletons(data_root, output_root, model_path='yolo11n-pose.pt'):
    """
    使用 YOLOv11 提取骨架关键点
    
    Args:
        data_root (str): CASIA-B 数据集根目录
        output_root (str): 骨架数据保存目录
        model_path (str): YOLOv11 模型路径
    """
    # 加载模型
    model = YOLO(model_path)
    
    data_path = Path(data_root)
    output_path = Path(output_root)
    
    # 遍历 CASIA-B 目录结构: Subject / WalkingStatus / View / Images
    # 假设结构: 001/nm-01/090/001.png
    
    subjects = sorted([d for d in data_path.iterdir() if d.is_dir()])
    
    for subject in tqdm(subjects, desc="Processing Subjects"):
        for status in sorted([d for d in subject.iterdir() if d.is_dir()]):
            for view in sorted([d for d in status.iterdir() if d.is_dir()]):
                
                save_dir = output_path / subject.name / status.name / view.name
                save_dir.mkdir(parents=True, exist_ok=True)
                
                image_files = sorted(list(view.glob('*.png')) + list(view.glob('*.jpg')))
                
                skeletons = []
                
                for img_file in image_files:
                    # 运行推理
                    results = model(str(img_file), verbose=False)
                    
                    # 获取关键点 (1, 17, 3) -> (x, y, conf)
                    # 如果检测到多个人，这里简单取第一个，或者根据置信度/面积筛选
                    if results[0].keypoints is not None and results[0].keypoints.data.shape[1] > 0:
                        kpts = results[0].keypoints.data[0].cpu().numpy() # (17, 3)
                    else:
                        kpts = np.zeros((17, 3)) # 未检测到
                        
                    skeletons.append(kpts)
                
                # 保存为 numpy 数组 (T, 17, 3)
                skeletons = np.array(skeletons)
                np.save(save_dir / 'skeletons.npy', skeletons)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='Path to CASIA-B dataset')
    parser.add_argument('--output_root', type=str, default='data/skeletons', help='Path to save skeletons')
    parser.add_argument('--model', type=str, default='yolo11n-pose.pt', help='YOLOv11 pose model path')
    args = parser.parse_args()
    
    extract_skeletons(args.data_root, args.output_root, args.model)
