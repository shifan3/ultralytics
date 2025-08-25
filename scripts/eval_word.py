
def draw_box(image, _4pts_box, color):
    draw = ImageDraw.Draw(image)
    x1, y1 = _4pts_box[0]
    x2, y2 = _4pts_box[1]
    x3, y3 = _4pts_box[2]
    x4, y4 = _4pts_box[3]
    draw.line([(x1, y1), (x2, y2)], width=2, fill=color)
    draw.line([(x2, y2), (x3, y3)], width=2, fill=color)
    draw.line([(x3, y3), (x4, y4)], width=2, fill=color)
    draw.line([(x4, y4), (x1, y1)], width=2, fill=color)


def x_cut(x_ranges, p1, p2):
    if p1 == p2:
        return x_ranges
    assert p1 < p2
    changed = True
    while changed:
        changed = False
        for i, (x1, x2) in enumerate(x_ranges):
            assert x1 < x2
            if p1 < x1 < p2 <= x2:
                x_ranges[i] = (p2, x2)
                changed = True
            elif x1 <= p1 <= p2 <= x2:
                x_ranges = x_ranges[:i] + [(x1, p1), (p2, x2)] + x_ranges[i+1:]
                changed = True
            elif x1 <= p1 < x2 < p2:
                x_ranges[i] = (x1, p1)
                changed = True
            elif p1 <= x1 <= x2 <= p2:
                x_ranges = x_ranges[:i] + x_ranges[i+1:]
                changed = True

            if changed:
                x_ranges = [(x1, x2) for x1, x2 in x_ranges if x1 != x2]
                x_ranges.sort(key = lambda x:x[0])
                break
    return x_ranges


from ultralytics import YOLO, RTDETR
from PIL import Image, ImageDraw, ImageFont

import json
from glob import glob
import cv2
import os
from tqdm import tqdm
import sys
import shutil
sys.path.append(os.getcwd())
from common import cls_to_color, classes_mapping
from rotated_rect_utils import (_4points_to_lefttop_rightbottom_theta, 
                                rotated_rect_contains_ratio, 
                                lefttop_rightbottom_theta_to_4points,
                                rotated_rect_contains)

model = YOLO(sys.argv[1])
with open('/mnt/data5/datasets/快读鸭英语点读数据/20250408纯单词表数据/val.txt', 'r', encoding='utf-8') as f:
    files = [line.strip() for line in f]

total_score = 0
total_count = 0
bad_cases = []
classes_mapping = [1, 3, 4, 5, 7, 14, 8, 6, 15, 11, 9, 10, 2]
allowed_classes = [classes_mapping[8]]
os.system('rm -rf data/bad_cases/ && mkdir -p data/bad_cases/')
os.system('rm -rf data/all_cases/ && mkdir -p data/all_cases/')
for file in tqdm(list(files)):
    #if '25A00ED80B074FF19536ADC4BE1BF811.jpg' not in file:
    #    continue
    file_score = 0
    file_count = 0
    json_file = file.replace('.jpg', '.json').replace('.png', '.json')
    with open(json_file, 'r') as f:
        data = json.load(f)
    label_regions = [region for region in data['regions'] if region['cls'] == classes_mapping[8]]
    label_regions.sort(key=lambda x : x['region'][0] + x['region'][1] * 5)
    for i, label_region in enumerate(label_regions):
        label_region['id'] = i
        label_region['box'] = label_region['region'] + [label_region['rotation']]
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
    result = model.predict(image, imgsz=1280, conf=0.1, iou=0.25, device="0", max_det=5000)[0]
    
    names = result.names
    obb = result.obb
    cls = obb.cls.tolist()
    conf = obb.conf.tolist()
    boxes = obb.xyxyxyxy.tolist()
    predicted = []
    for i in range(len(cls)):
        _4pts_box = boxes[i]
        for pt in _4pts_box:
            pt[0] = min(max(0, pt[0]), image.width)
            pt[1] = min(max(0, pt[1]), image.height)
        box = _4points_to_lefttop_rightbottom_theta(_4pts_box)
        color = cls_to_color[cls[i]%13]
        cls_1 = classes_mapping[int(cls[i] % 13)]
        if cls_1 in allowed_classes:
            predicted.append({
                'cls' : cls_1,
                'region' : list(box[:4]),
                'rotation' : box[4],
                'box' : box,
                '4pts_box' : _4pts_box,
                'score' : conf[i],
                'color' : (color[2], color[1], color[0]),
            })
            
        # color = (0, 0, 255)
    print(len(predicted), len(label_regions))
    
    
    removed_idxes = []
    for i in range(len(predicted)):
        for j in range(len(predicted)):
            if i == j:
                continue
            #if predicted[i]['cls'] != predicted[j]['cls']:
            #    continue
            if rotated_rect_contains(predicted[i]['box'], predicted[j]['box']):
                if predicted[i]['score'] >= predicted[j]['score']:
                    removed_idxes.append(j)
    
    predicted = [predicted[i] for i in range(len(predicted)) if i not in removed_idxes]

    
    
    ori_predicted = predicted
    label_regions = [region for region in label_regions if region['cls'] in allowed_classes]
    for label_region in label_regions:
        label_box = label_region['box']
        match = (None, 0, 0, 0)
        for i in range(len(predicted)):
            predicted_item = predicted[i]
            color = predicted_item['color']
            r1 = rotated_rect_contains_ratio(label_box, predicted_item['box'])
            r2 = rotated_rect_contains_ratio(predicted_item['box'], label_box)

            if (min(r1, r2)) > match[1] and min(r1, r2) > 0.8:
                match = (predicted[i], min(r1, r2), r1, r2)
        _4pts_box = None
        if match[0] is not None:
            label_region['matched'] = match[0]
            match[0]['matched'] = label_region
            predicted_item = match[0]
            r1, r2 = match[2:]
            box = predicted_item['box'] 
            _4pts_box = predicted_item['4pts_box']
            h = label_box[3] - label_box[1]
            #x_diff = max(0, abs(box[0] - label_box[0]) + abs(label_box[2] - box[2]))
            #y_diff = abs(label_box[1] - box[1]) + abs(label_box[3] - box[3])
            x_ranges = x_cut([(label_box[0], label_box[2])], box[0], box[2])
            remain = sum([abs(x2-x1) for x1, x2 in x_ranges])
            score = 1 - remain / (label_box[2] - label_box[0])
            if score < 0.98:
                if _4pts_box is not None:
                    draw_box(image, _4pts_box, color)
                label_4pts_box = lefttop_rightbottom_theta_to_4points(label_box)
                draw_box(image, label_4pts_box, 'red')
                #print('T1', label_region['id'])
            else:
                pass
                #print('M1', label_region['id'])
            assert score <= 1
            total_score += score
            total_count += 1
            file_count += 1
            file_score += score
            
    predicted = [predicted_item for predicted_item in predicted if 'matched' not in predicted_item]
    label_regions = [label_region for label_region in label_regions if 'matched' not in label_region]
    
    for label_region in label_regions:
        x_ranges = [(label_region['region'][0], label_region['region'][2])]
        total_x = sum([abs(x2-x1) for x1, x2 in x_ranges])
        contained = []
        for predicted_item in predicted:
            if rotated_rect_contains(label_region['box'], predicted_item['box']):
                l_x1 = predicted_item['region'][0]
                l_x2 = predicted_item['region'][2]
                x_ranges = x_cut(x_ranges, l_x1, l_x2)
                predicted_item['matched'] = label_region
                contained.append(predicted_item)
        if contained:
            label_region['matched'] = contained
            remain = sum([abs(x2-x1) for x1, x2 in x_ranges])
            score = 1 - remain / total_x
            assert score <= 1
            total_score += score
            total_count += 1
            file_score += score
            file_count += 1
            if score < 0.95:
                for contained_item in contained:
                    draw_box(image, contained_item['4pts_box'], 'blue')
                label_4pts_box = lefttop_rightbottom_theta_to_4points(label_region['box'])
                draw_box(image, label_4pts_box, 'red')
                #print('B1', label_region['id'])
            else:
                pass
                #print('A1', label_region['id'])
    
    predicted = [predicted_item for predicted_item in predicted if 'matched' not in predicted_item]
    label_regions = [label_region for label_region in label_regions if 'matched' not in label_region]

    del label_region
    for predicted_item in predicted:
        x_ranges = []
        contained = []
        for label_region in label_regions:
            if rotated_rect_contains(predicted_item['box'], label_region['box']):
                l_x1 = label_region['region'][0]
                l_x2 = label_region['region'][2]
                x_ranges.append((l_x1, l_x2))
                label_region['matched'] = predicted_item
                contained.append(label_region)
        if contained:
            total_x = sum([abs(x2-x1) for x1, x2 in x_ranges])
            x_ranges = x_cut(x_ranges, predicted_item['region'][0], predicted_item['region'][2])
            predicted_item['matched'] = contained
            remain = sum([abs(x2-x1) for x1, x2 in x_ranges])
            score = 1 - remain / total_x
            assert score <= 1
            total_score += score
            total_count += 1
            file_score += score
            file_count += 1
            if score < 0.95:
                for contained_item in contained:
                    label_4pts_box = lefttop_rightbottom_theta_to_4points(contained_item['box'])
                    draw_box(image, label_4pts_box, 'red')
                
                draw_box(image, predicted_item['4pts_box'], 'blue')
                #print('B2', [r['id'] for r in contained])
            else:
                pass
                #print('A2', [r['id'] for r in contained])
    predicted = [predicted_item for predicted_item in predicted if 'matched' not in predicted_item]
    label_regions = [label_region for label_region in label_regions if 'matched' not in label_region]

    for label_region in label_regions:
        score = 0
        total_count += 1
        file_count += 1
        label_4pts_box = lefttop_rightbottom_theta_to_4points(label_region['box'])
        draw_box(image, label_4pts_box, 'red')
        #print('Q', label_region['id'])

    for predict_item in predicted:
        _4pts_box = lefttop_rightbottom_theta_to_4points(predict_item['box'])
        #draw_box(image, _4pts_box, 'blue')
    
    
    print(file, file_score, file_count, file_score / file_count if file_count > 0 else 0)
    
        

    image_predicted = Image.open(file)
    for predicted_item in ori_predicted:
        _4pts_box = predicted_item['4pts_box']
        color = predicted_item['color']
        
        score = predicted_item['score']
        draw_box(image_predicted, _4pts_box, color)
        if score < 0.25:
            draw = ImageDraw.Draw(image_predicted)
            font = ImageFont.load_default()
            text = f"{score}"
            text_position = (_4pts_box[0][0], _4pts_box[0][1])
            draw.text(text_position, text, fill=color, font=font)
        
    if file_score / file_count < 0.9:
        image.save('data/bad_cases/' + os.path.basename(file))
        image_predicted.save('data/bad_cases/' + os.path.basename(file).replace('.jpg', '.predicted.jpg').replace('.png', '.predicted.jpg'))
        shutil.copy(file, 'data/bad_cases/' + os.path.basename(file) + '.ori.jpg')
    if file_score / file_count < 0.999:
        image.save('data/all_cases/' + os.path.basename(file))
        image_predicted.save('data/all_cases/' + os.path.basename(file).replace('.jpg', '.predicted.jpg'))
        shutil.copy(file, 'data/all_cases/' + os.path.basename(file) + '.ori.jpg')
    
print(total_score, total_count, total_score / total_count)





