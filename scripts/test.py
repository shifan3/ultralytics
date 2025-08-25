
from glob import glob
import cv2
import os
import sys
sys.path.append(os.getcwd())
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from common import cls_to_color, classes_mapping, resize_keep_ratio

model = YOLO(sys.argv[1])
os.system('rm -rf data/test_results')
os.system('mkdir -p data/test_results')
files = glob('/mnt/data5/english_reader/configs/eng/*.jpg') + glob('/mnt/data5/english_reader/configs/eng/*.png')
for file in files:
    print(file)
    image = Image.open(file)
    # 根据EXIF信息旋转图片
    try:
        exif = image._getexif()
        if exif:
            orientation = exif.get(274)  # 274 对应 Orientation 标签
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
    except:
        pass
    #image = image.resize((1024, 1024))
    result = model.predict(image, imgsz=1280, conf=0.25, iou=0.25, device="0", max_det=1000, verbose = False, classes=[8])[0]
    
    names = result.names
    obb = result.obb
    cls = obb.cls.tolist()
    conf = obb.conf.tolist()
    boxes = obb.xyxyxyxy.tolist()
    for i in range(len(cls)):
        box = boxes[i]
        for pt in box:
            pt[0] = min(max(0, pt[0]), image.width)
            pt[1] = min(max(0, pt[1]), image.height)
        x1, y1 = box[0]
        x2, y2 = box[1]
        x3, y3 = box[2]
        x4, y4 = box[3]


        color = cls_to_color[cls[i]%13]
        #if cls[i] not in [8]: #[4, 13]:
        #    continue
        # color = (0, 0, 255)
        #color = 'blue'
        color = (color[2], color[1], color[0])
        #if cls[i] not in [4, 13]:
        #    continue
        draw = ImageDraw.Draw(image)
        draw.line([(x1, y1), (x2, y2)], width=1, fill=color)
        draw.line([(x2, y2), (x3, y3)], width=1, fill=color)
        draw.line([(x3, y3), (x4, y4)], width=1, fill=color)
        draw.line([(x4, y4), (x1, y1)], width=1, fill=color)
    image.save('data/test_results/' + os.path.basename(file) + '.result.png')
        



