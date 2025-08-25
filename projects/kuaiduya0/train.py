import sys
import os
sys.path.append(os.getcwd())
from ultralytics import YOLO, RTDETR
import torch
from io import BytesIO
import pickle
import base64
curr_dir = os.path.dirname(os.path.abspath(__file__))

model = YOLO("./models/obb_pretrain/best.pt")

project_name = os.path.basename(curr_dir).split(".")[0]

results = model.train(
        data=f"{curr_dir}/kuaiduya.yaml", 
        epochs=200, 
        imgsz=1280, 
        device=list(range(torch.cuda.device_count())), 
        project=project_name, 
        recreate_cache=False)
