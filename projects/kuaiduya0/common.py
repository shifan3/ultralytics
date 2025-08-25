import math
from enum import Enum
import sys
import os
sys.path.append(os.getcwd())
from projects import ProjectBase
_cls_to_color = {
            1: (255, 0, 0),  # 蓝色 文本
            2: (203, 192, 255),  # 粉色 手写
            3: (0, 0, 255),  # 红色 边框
            4: (0, 255, 0),  # 绿色 音乐
            5: (0, 0, 255),  # 红色 对话框
            6: (255, 255, 0),  # 青色 单词+发音
            7: (0, 0, 255),  # 红色 块
            8: (128, 0, 128),  # 紫色 序号
            9: (45, 82, 160),  # 棕色 选项框
            10: (130, 0, 75),  # 靛蓝色 选项
            11: (0, 255, 255),  # 黄色 发音
            14: (0, 165, 255),  # 橙色 表格
            15: (128, 128, 0),  # 蓝绿色 单词
            16: (0, 255, 255),  # 橙色 单行
        }

def cls_to_color(cls):
    return _cls_to_color[cls]

classes_mapping = [1, 3, 4, 5, 7, 14, 8, 6, 15, 11, 9, 10, 2, 16]
classes_mapping = {v: i for i, v in enumerate(classes_mapping)}
rev_classes_mapping = {v: k for k, v in classes_mapping.items()}



class CLS(Enum):
    TEXT = 1
    HANDWRITING = 2
    BORDER = 3
    MUSIC = 4
    DIALOG = 5
    WORD_PHONETIC = 6
    BLOCK = 7
    ORDER = 8
    OPTION = 9
    OPTION_2 = 10
    PHONETIC = 11
    TABLE = 14
    WORD = 15
    SINGLE_LINE = 16    


def cls_to_name(cls):
    return CLS(cls).name

class Project(ProjectBase):
    def cls_to_color(self, cls):
        return cls_to_color(cls)

    def cls_to_name(self, cls):
        return CLS(cls).name
    
    def cls_to_raw(self, cls):
        return classes_mapping[cls]

    def cls_selector(self, cls):
        if cls in classes_mapping:
            return classes_mapping[cls]
        else:
            return None
    
    def raw_to_cls(self, raw):
        return rev_classes_mapping[raw]
    
    def image_size(self):
        return 1024

    def no_merge_cls(self):
        return [CLS.BLOCK]
