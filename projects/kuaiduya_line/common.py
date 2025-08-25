import math
from enum import Enum
import sys
import os
sys.path.append(os.getcwd())
from projects import ProjectBase


_cls_to_color = {
            1: (255, 0, 0),  # 蓝色 句子
            2: (203, 192, 255),  # 粉色 手写
            3: (0, 0, 255),  # 红色 边框
            4: (0, 255, 0),  # 绿色 文本
            5: (0, 0, 255),  # 红色 对话框
            7: (0, 0, 255),  # 红色 块
            8: (128, 0, 128),  # 紫色 序号
            10: (255, 0, 0),  # 蓝色 句子
            11: (255, 0, 0),  # 蓝色 句子
            14: (0, 165, 255),  # 橙色 表格
            15: (255, 0, 0),  # 蓝色 句子
        }

def cls_to_color(cls):
    return _cls_to_color[cls]

cls_map = {4: 0, 2: 1, 5: 2, 7: 3, 3: 4, 1: 5, 8: 6, 15: 5, 11: 5, 10: 5, 14: 7}
rev_cls_map = {v: k for k, v in cls_map.items()}




class CLS(Enum):
    TEXT = [1, 10, 11, 15]
    HANDWRITING = 2
    BORDER = 3
    LINE = 4
    DIALOG = 5
    BLOCK = 7
    ORDER = 8
    TABLE = 14 

class Project(ProjectBase):
    def cls_to_color(self, cls):
        return cls_to_color(cls)

    def cls_to_name(self, cls):
        
        if cls in [1, 10, 11, 15]:
            return CLS.TEXT.name
        return CLS(cls).name

    def cls_selector(cls):
        if cls in cls_map:
            return cls_map[cls]
        else:
            return None
    
    def cls_to_raw(self, cls):
        return cls_map[cls]
    
    def raw_to_cls(self, raw):
        return rev_cls_map[raw]
    
    def image_size(self):
        return 1536

    def no_merge_cls(self):
        return [CLS.BLOCK, CLS.DIALOG, CLS.TABLE, CLS.ORDER, CLS.LINE, CLS.BORDER]

    def all_cls(self):
        return [CLS.TEXT, CLS.HANDWRITING, CLS.BORDER, CLS.LINE, CLS.DIALOG, CLS.BLOCK, CLS.ORDER, CLS.TABLE]
