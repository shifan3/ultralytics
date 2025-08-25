import sys
import os
sys.path.append(os.getcwd())
from ultralytics import YOLO, RTDETR
import torch
from common import Project
curr_dir = os.path.dirname(os.path.abspath(__file__))

model = YOLO("/mnt/data5/ultralytics/models/obb_pretrained_2/weights/best.pt")

project_name = os.path.basename(curr_dir).split(".")[0]
project = Project()
"""
box: 10 # (float) box loss gain
cls: 0.5 # (float) cls loss gain (scale with pixels)
dfl: 2.0 # (float) dfl loss gain"""
results = model.train(
        data=f"{curr_dir}/kuaiduya.yaml", 
        epochs=200, 
        imgsz=project.image_size(), 
        device=list(range(torch.cuda.device_count())), 
        project=project_name, 
        lrf=0.0001,
        lr0=0.0001,
        optimizer='AdamW',
        save_period=10,
        recreate_cache=False)
