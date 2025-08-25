

import os
import sys
sys.path.append(os.getcwd())
from rotated_rect_utils import (points4_to_lefttop_rightbottom_theta, 
                                rotated_rect_contains_ratio, 
                                lefttop_rightbottom_theta_to_4points,
                                rotated_rect_contains)
from PIL import Image, ImageDraw, ImageFont
import json
from ultralytics import YOLO, RTDETR
from glob import glob
import cv2
import os
from tqdm import tqdm
import sys
import shutil
from projects import ProjectBase as Project
from collections import defaultdict


#baseline allow=[1, 15, 11, 10, 2]: 9566.931353310752 10179 0.939869471786104
#baseline allow=[7]: 403.3553742935509 511 0.7893451551732894







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


def draw_text(image, text, text_position, color, font_size=16):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font='/mnt/data5/kuaiduya/configs/fonts/msyh.ttc', size=font_size)
    draw.text(text_position, text, fill=color, font=font)


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



def test_det(checkpoint_file, test_files, allowed_classes, testcase_project: Project, model_project: Project, output_dir: str = 'data/', slient=False, device = "0"):

    assert all(isinstance(i, str) for i in allowed_classes)
    no_merge_cls = [model_project.cls_to_name(cls) for cls in model_project.no_merge_cls()]
    model = YOLO(checkpoint_file)
    #files = glob('/mnt/data5/datasets/快读鸭英语点读det测试集/**/*.jpg', recursive=True)
    total_score = 0
    total_count = 0
    total_score_by_cls = defaultdict(int)
    total_count_by_cls = defaultdict(int)

    #allowed_classes = [7]#[classes_mapping[i1] for i1 in [0,8,9,11,12]]
    #allowed_classes = [model_project.cls_to_name(cls) for cls in allowed_classes]
    if output_dir:
        os.system(f'rm -rf {output_dir}/bad_cases/ && mkdir -p {output_dir}/bad_cases/')
        os.system(f'rm -rf {output_dir}/all_cases/ && mkdir -p {output_dir}/all_cases/')
    for file in tqdm(list(test_files), disable=slient):
        #if '25A00ED80B074FF19536ADC4BE1BF811.jpg' not in file:
        #    continue
        file_score = 0
        file_count = 0
        file_score_by_cls = defaultdict(int)
        file_count_by_cls = defaultdict(int)
        txt_file = file.replace('.jpg', '.txt')
        with open(txt_file, 'r') as f:
            data = json.load(f)
        label_regions = data['regions']
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

        expanded_image = Image.new('RGB', (image.width * 2, image.height * 2), (255, 255, 255))
        expanded_image.paste(image, (0, 0))
        #image = expanded_image
        result = model.predict(image, imgsz=model_project.image_size(), conf=0.1, iou=0.25, device=device, max_det=1000, verbose=False)[0]
        
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
            box = points4_to_lefttop_rightbottom_theta(_4pts_box)
            
            cls_1 = model_project.raw_to_cls(int(cls[i]))
            assert cls_1 is not None, f'mapper error, {cls[i]}'
            color = model_project.cls_to_color(cls_1)
            if model_project.cls_to_name(cls_1) in allowed_classes:
                predicted.append({
                    'cls' : cls_1,
                    'region' : list(box[:4]),
                    'rotation' : box[4],
                    'box' : box,
                    '4pts_box' : _4pts_box,
                    'score' : conf[i],
                    'color' : (color[2], color[1], color[0]),
                })
            #else:
            #    print(model_project.cls_to_name(cls_1), 'not in', allowed_classes)
                
            # color = (0, 0, 255)
        
        removed_idxes = []
        for i in range(len(predicted)):
            for j in range(len(predicted)):
                if i == j:
                    continue
                if predicted[i]['cls'] != predicted[j]['cls']:
                    continue
                if rotated_rect_contains(predicted[i]['box'], predicted[j]['box']):
                    if predicted[i]['score'] >= predicted[j]['score']:
                        removed_idxes.append(j)
        
        predicted = [predicted[i] for i in range(len(predicted)) if i not in removed_idxes]
        
        ori_predicted = predicted
        label_regions = [region for region in label_regions if testcase_project.cls_to_name(region['cls']) in allowed_classes]
        for label_region in label_regions:
            
            rs = []
            label_box = label_region['box']
            match = (None, 0, 0, 0)
            for predicted_item in predicted:
                if model_project.cls_to_name(predicted_item['cls']) != testcase_project.cls_to_name(label_region['cls']):
                    #print('MATCH', model_project.cls_to_name(predicted_item['cls']), '!=', testcase_project.cls_to_name(label_region['cls']))
                    continue
                color = predicted_item['color']
                r1 = rotated_rect_contains_ratio(label_box, predicted_item['box'])
                r2 = rotated_rect_contains_ratio(predicted_item['box'], label_box)
                rs.append(min(r1, r2))
                if (min(r1, r2)) > match[1] and min(r1, r2) > 0.7:
                    
                    match = (predicted_item, min(r1, r2), r1, r2)
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
                if score < 0.9 and output_dir:
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
                file_score_by_cls[testcase_project.cls_to_name(label_region['cls'])] += score
                file_count_by_cls[testcase_project.cls_to_name(label_region['cls'])] += 1
                total_score_by_cls[testcase_project.cls_to_name(label_region['cls'])] += score
                total_count_by_cls[testcase_project.cls_to_name(label_region['cls'])] += 1
            #else:
            #    print(testcase_project.cls_to_name(label_region['cls']), max(rs))
            #    draw_text(image, testcase_project.cls_to_name(label_region['cls']) + ' ' + str(max(rs)), label_box[:2], 'yellow')
        

        predicted = [predicted_item for predicted_item in predicted if 'matched' not in predicted_item]

        label_regions = [label_region for label_region in label_regions if 'matched' not in label_region]
        

        for label_region in label_regions:
            if testcase_project.cls_to_name(label_region['cls']) in no_merge_cls:
                continue
            x_ranges = [(label_region['region'][0], label_region['region'][2])]
            total_x = sum([abs(x2-x1) for x1, x2 in x_ranges])
            contained = []
            for predicted_item in predicted:
                if testcase_project.cls_to_name(predicted_item['cls']) in no_merge_cls:
                    continue
                if model_project.cls_to_name(predicted_item['cls']) != testcase_project.cls_to_name(label_region['cls']):
                    continue
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
                file_score_by_cls[testcase_project.cls_to_name(label_region['cls'])] += score
                file_count_by_cls[testcase_project.cls_to_name(label_region['cls'])] += 1
                total_score_by_cls[testcase_project.cls_to_name(label_region['cls'])] += score
                total_count_by_cls[testcase_project.cls_to_name(label_region['cls'])] += 1
                if score < 0.9 and output_dir:
                    for contained_item in contained:
                        draw_box(image, contained_item['4pts_box'], 'blue')
                    label_4pts_box = lefttop_rightbottom_theta_to_4points(label_region['box'])
                    draw_box(image, label_4pts_box, 'purple')
                    draw_text(image, testcase_project.cls_to_name(label_region['cls']), label_4pts_box[0], 'purple')
                    #print('B1', label_region['id'])
                else:
                    pass
                    #print('A1', label_region['id'])
        
        predicted = [predicted_item for predicted_item in predicted if 'matched' not in predicted_item]
        label_regions = [label_region for label_region in label_regions if 'matched' not in label_region]

        del label_region
        for predicted_item in predicted:
            if model_project.cls_to_name(predicted_item['cls']) in no_merge_cls:
                continue
            
            x_ranges = []
            contained = []
            for label_region in label_regions:
                if testcase_project.cls_to_name(label_region['cls']) in no_merge_cls:
                    continue
                if model_project.cls_to_name(predicted_item['cls']) != testcase_project.cls_to_name(label_region['cls']):
                    continue
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
                file_score_by_cls[testcase_project.cls_to_name(predicted_item['cls'])] += score
                file_count_by_cls[testcase_project.cls_to_name(predicted_item['cls'])] += 1
                total_score_by_cls[testcase_project.cls_to_name(predicted_item['cls'])] += score
                total_count_by_cls[testcase_project.cls_to_name(predicted_item['cls'])] += 1
                if score < 0.95 and output_dir:
                    for contained_item in contained:
                        label_4pts_box = lefttop_rightbottom_theta_to_4points(contained_item['box'])
                        draw_box(image, label_4pts_box, 'cyan')
                        draw_text(image, testcase_project.cls_to_name(contained_item['cls']), label_4pts_box[0], 'cyan')
                    
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
            file_count_by_cls[testcase_project.cls_to_name(label_region['cls'])] += 1
            total_count_by_cls[testcase_project.cls_to_name(label_region['cls'])] += 1
            if output_dir:
                label_4pts_box = lefttop_rightbottom_theta_to_4points(label_region['box'])
                draw_box(image, label_4pts_box, 'pink')
                draw_text(image, testcase_project.cls_to_name(label_region['cls']), label_4pts_box[0], 'pink')
            #print('Q', label_region['id'])

        for predict_item in predicted:
            if output_dir:
                _4pts_box = lefttop_rightbottom_theta_to_4points(predict_item['box'])
                draw_box(image, _4pts_box, 'orange')
                draw_text(image, model_project.cls_to_name(predict_item['cls']), _4pts_box[0], 'orange')
            total_count += 1
            file_count += 1
            file_count_by_cls[model_project.cls_to_name(predict_item['cls'])] += 1
            total_count_by_cls[model_project.cls_to_name(predict_item['cls'])] += 1
        
        file_avg_score = file_score / file_count if file_count > 0 else 0
        if not slient:
            print(file, file_score, file_count, file_avg_score)
        
            
        if output_dir:
            image_predicted = Image.open(file)
            for predicted_item in ori_predicted:
                #predicted_item['4pts_box']
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
            draw_text(image, f'{file_avg_score}', (10, 10), 'red')
        
            if file_count > 0 and file_score / file_count < 0.9:
                image.save(f'{output_dir}/bad_cases/' + os.path.basename(file))
                image_predicted.save(f'{output_dir}/bad_cases/' + os.path.basename(file).replace('.jpg', '.predicted.jpg'))
                shutil.copy(file, f'{output_dir}/bad_cases/' + os.path.basename(file) + '.ori.jpg')
            #if file_count > 0 and file_score / file_count < 0.999:
            image.save(f'{output_dir}/all_cases/' + os.path.basename(file))
            image_predicted.save(f'{output_dir}/all_cases/' + os.path.basename(file).replace('.jpg', '.predicted.jpg'))
            shutil.copy(file, f'{output_dir}/all_cases/' + os.path.basename(file) + '.ori.jpg')
        
    if not slient:
        print(total_score, total_count, total_score / total_count)
        for cls in total_score_by_cls:
            print(cls, total_score_by_cls[cls], total_count_by_cls[cls], total_score_by_cls[cls] / total_count_by_cls[cls] if total_count_by_cls[cls] > 0 else 0)
    
    return {
        'total_score' : total_score,
        'total_count' : total_count,
        'total_score_by_cls' : total_score_by_cls,
        'total_count_by_cls' : total_count_by_cls
    }





