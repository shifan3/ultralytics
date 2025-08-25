import os
import sys
sys.path.append(os.getcwd())
from argparse import ArgumentParser
from eval_utils import test_det
from glob import glob
from projects.kuaiduya0.common import Project as Kuaiduya0Project
from projects.kuaiduya1.common import Project as Kuaiduya1Project, CLS
import xlsxwriter
from tqdm import tqdm
import torch
import multiprocessing as mp
import json
parser = ArgumentParser()
parser.add_argument('model_dir', type=str)
parser.add_argument('--output', '-o', type=str, default=None)
args = parser.parse_args()

ngpus = torch.cuda.device_count()
#prod /mnt/ceph2/kuaiduya-engine/models/detection/line.pt
#baseline 4170.780418627358 8295 0.5028065604131836
lock = mp.Lock()
counter = mp.Value('i', 0)
def safe_init_worker():
    global _worker_id
    with lock:
        _worker_id = counter.value
        counter.value += 1
"""
TOTAL 3117.5787738291106 4102 0.7600143280909583
BLOCK 161.34855202214231 371 0.4349017574720817
TEXT 2537.381837725799 2944 0.8618824177057741
ORDER 152.39229386383423 330 0.46179482989040677
BORDER 116.29252677229215 264 0.44050199534959145
DIALOG 79.79248677353979 106 0.7527593091843376
OPTION_2 67.46556347877531 68 0.9921406393937545
TABLE 2.9055131927188045 9 0.3228347991909783


"""
base_line = {
    'TOTAL' : 0.7600143280909583,
    'BLOCK' : 0.4349017574720817,
    'TEXT' : 0.8618824177057741,
    'ORDER' : 0.46179482989040677,
    'BORDER' : 0.44050199534959145,
    'DIALOG' : 0.7527593091843376,
    'OPTION_2' : 0.9921406393937545,
    'TABLE' : 0.3228347991909783,
    'HANDWRITING' : 0.0,
    'LINE' : 0.0,
}



test_files = []

for test_file in glob('/mnt/data5/datasets/快读鸭英语点读det测试集/**/*.jpg', recursive=True):
    json_file = test_file.replace('.jpg', '.txt')
    with open(json_file, 'r') as f:
        data = json.load(f)

    if any(region['cls'] in [15, 11, 6] for region in data['regions']):
        continue
    test_files.append(test_file)

allowed_classes = [CLS.TEXT, CLS.HANDWRITING, CLS.BLOCK, CLS.DIALOG, CLS.TABLE, CLS.ORDER, CLS.BORDER]
allowed_classes = [c.name for c in allowed_classes]

if args.output:
    workbook = xlsxwriter.Workbook(args.output)
    worksheet = workbook.add_worksheet()
    worksheet.write(0, 0, 'file')
    worksheet.write(0, 1, 'TOTAL')
    worksheet.write(0, 2, 'COMMENT')    
    for i, cls in enumerate(allowed_classes):
        worksheet.write(0, i + 3, cls)

def test_det_wrapper(ckpt_file):
    global _worker_id
    ret = test_det(ckpt_file, test_files, allowed_classes, testcase_project=Kuaiduya0Project(), model_project=Kuaiduya1Project(), output_dir=None, slient=True, device=str(_worker_id % ngpus))
    ret['ckpt_file'] = ckpt_file
    return ret

ckpt_files = glob(os.path.join(args.model_dir, '**/*.pt'), recursive=True)

best_score = (0, None)
with mp.Pool(ngpus, initializer=safe_init_worker) as pool:
    for row, ret in enumerate(tqdm(pool.imap_unordered(test_det_wrapper, ckpt_files), total=len(ckpt_files))):
        #ret = test_det(ckpt_file, test_files, allowed_classes, testcase_project=Kuaiduya0Project(), model_project=Kuaiduya1Project(), output_dir=None, slient=True)
        ckpt_file = ret['ckpt_file']
        total_score = ret['total_score'] / ret['total_count']
        score_by_cls = {
            'TOTAL' : total_score,
        }
        if total_score > best_score[0]:
            best_score = (total_score, ckpt_file)
        for cls in allowed_classes:
            score_by_cls[cls] = ret['total_score_by_cls'].get(cls, 0) / ret['total_count_by_cls'].get(cls, 0) if ret['total_count_by_cls'].get(cls, 0) > 0 else 0
        
        if all(score_by_cls[cls] >= base_line[cls] for cls in allowed_classes):
            comment = 'SURPASS'
        elif all(score_by_cls[cls] <= base_line[cls] for cls in allowed_classes):
            comment = 'FALL BEHIND'
        else:
            comment = 'MIXED'
        if not args.output:
            print(ckpt_file)
            print(f'TOTAL: {score_by_cls["TOTAL"]:.2%}')
            print(comment)
        else:
            worksheet.write(row + 1, 0, ckpt_file)
            worksheet.write(row + 1, 1, f'{score_by_cls["TOTAL"]:.2%}')
            worksheet.write(row + 1, 2, comment)
        for i, cls in enumerate(allowed_classes):
            if args.output:
                worksheet.write(row + 1, i + 3, score_by_cls[cls])
            else:
                print(cls, score_by_cls[cls])
print('best score', best_score[0], best_score[1])
workbook.close()








