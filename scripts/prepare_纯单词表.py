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
from common import classes_mapping

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

    files = glob('/mnt/data5/datasets/快读鸭英语点读数据/20250408纯单词表数据/**/*.jpg', recursive=True)
    files += glob('/mnt/data5/datasets/快读鸭英语点读数据/20250408纯单词表数据/**/*.png', recursive=True)
    
    random.shuffle(files)  
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
    for path in tqdm(files):
        with open(path[:-4]+'.json', encoding='utf-8') as f:
            data = json.load(f)
        try:
            img = cv2.imread(path)
            h, w = img.shape[:2]
        except:
            print(path)
            continue
        with open(path[:-4] + '.txt', 'w', encoding='utf-8') as f:
            for region in data['regions']:
                # if region['cls'] not in [1]:
                #     continue
                cls = classes_mapping.index(region['cls'])
                if cls not in [8]:
                    continue
                p1, p2, p3, p4 = lefttop_rightbottom_theta_to_4points(region['region'] + [region['rotation']])
                p1 = (p1[0] / w, p1[1] / h)
                p2 = (p2[0] / w, p2[1] / h)
                p3 = (p3[0] / w, p3[1] / h)
                p4 = (p4[0] / w, p4[1] / h)
                f.write('%d %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n' % (cls, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], p4[0], p4[1]))


    # files = glob('/mnt/ceph2/datasets/diandu/link2/**/*.jpg', recursive=True)
    # files += glob('/mnt/ceph2/datasets/diandu/link2/**/*.png', recursive=True)
    # random.seed(0)
    # random.shuffle(files)
    with open('/mnt/data5/datasets/快读鸭英语点读数据/20250408纯单词表数据/train.txt', 'w', encoding='utf-8') as f:
        for path in files[10:]:
            f.write(path+'\n')
    with open('/mnt/data5/datasets/快读鸭英语点读数据/20250408纯单词表数据/val.txt', 'w', encoding='utf-8') as f:
        for path in files[:10]:
            f.write(path+'\n')


if __name__ == '__main__':
    process_diandu_link2()
    