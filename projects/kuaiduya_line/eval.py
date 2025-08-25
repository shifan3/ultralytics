import os
import sys
sys.path.append(os.getcwd())
from argparse import ArgumentParser
from eval_utils import test_det
from glob import glob
from projects.kuaiduya0.common import Project as Kuaiduya0Project
from projects.kuaiduya1.common import Project as Kuaiduya1Project, CLS
import json
parser = ArgumentParser()
parser.add_argument('model', type=str)
parser.add_argument('--output_dir', type=str, default=None)
parser.add_argument('--classes', type=str, default=None)
args = parser.parse_args()

#prod /mnt/ceph2/kuaiduya-engine/models/detection/line.pt
#baseline 4170.780418627358 8295 0.5028065604131836


test_files = []

for test_file in glob('/mnt/data5/datasets/快读鸭英语点读det测试集/**/*.jpg', recursive=True):
    json_file = test_file.replace('.jpg', '.txt')
    with open(json_file, 'r') as f:
        data = json.load(f)

    if any(region['cls'] in [15, 11, 6] for region in data['regions']):
        continue
    test_files.append(test_file)

if args.classes is None:
    allowed_classes = [CLS.TEXT, CLS.HANDWRITING, CLS.BLOCK, CLS.DIALOG, CLS.TABLE, CLS.ORDER, CLS.BORDER]
    allowed_classes = [c.name for c in allowed_classes]
else:
    allowed_classes = args.classes.split(',')

test_det(args.model, test_files, allowed_classes, testcase_project=Kuaiduya0Project(), model_project=Kuaiduya1Project(), output_dir=args.output_dir)