import json
import os
import cv2
import shutil
import random
import numpy as np
from glob import glob
from tqdm import tqdm
import sys
sys.path.append(os.getcwd())

cls_map = {1:0, 2:1, 3:2, 4:3, 5:3, 6:3, 7:3, 8:3, 9:3, 10:0, 12:4, 13:5, 14:6, 15:7, 16:8, 17:0, 18:9, 19:5}

def lefttop_rightbottom_theta_to_4points(region):
    x1, y1, x2, y2, theta = region
    points = []
    points.append((x1, y1, 1))
    points.append((x2, y1, 1))
    points.append((x2, y2, 1))
    points.append((x1, y2, 1))

    M = cv2.getRotationMatrix2D((x1, y1), - theta, 1)
    Mt = np.transpose(M)
    roated_points = np.matmul(points, Mt)
    ret = []
    for i in range(roated_points.shape[0]):
        ret.append(tuple(roated_points[i]))
    return ret


def process_diandu_link2():
    # files = glob('/mnt/ceph2/datasets/diandu/link2/20250312单词表数据/**/*.txt', recursive=True)
    # for path in tqdm(files):
    #     shutil.move(path, path.replace('.txt', '.json'))
    # exit(0)
    for file in ['/mnt/ceph2/cxy/datasets/word_det/SJB_DET_20240701/images_with_labels/train.txt', '/mnt/ceph2/cxy/datasets/word_det/SJB_DET_20240701/images_with_labels/val.txt']:
        for path in tqdm(list(open(file, 'r', encoding='utf-8'))):
            img_path = path.strip()
            path = path.strip().replace('.jpg', '.txt').replace('.png', '.txt')
            json_path = path.replace('.txt', '.json')
            if not os.path.exists(json_path) and os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    data = f.read().strip()
                if not data[0].isdigit():
                    shutil.move(path, json_path)
            try:
                img = cv2.imread(img_path)
                h, w = img.shape[:2]
            except:
                print('skip', img_path)
                continue
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            with open(path + '.tmp', 'w', encoding='utf-8') as f:
                for region in data['regions']:
                    if region['cls'] == 11:
                        continue
                    cls = cls_map[region['cls']]
                    p1, p2, p3, p4 = lefttop_rightbottom_theta_to_4points(region['region'] + [region['rotation']])
                    p1 = (p1[0] / w, p1[1] / h)
                    p2 = (p2[0] / w, p2[1] / h)
                    p3 = (p3[0] / w, p3[1] / h)
                    p4 = (p4[0] / w, p4[1] / h)
                    f.write('%d %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n' % (cls, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], p4[0], p4[1]))
            shutil.move(path + '.tmp', path)



    '''
    0: sentence
    1: border
    2: music
    3: dialog
    4: block
    5: table
    6: order
    7: word+symbol
    8: word
    9: symbol
    10: options
    11: option
    12: handwriting
    '''
    


if __name__ == '__main__':
    process_diandu_link2()
    