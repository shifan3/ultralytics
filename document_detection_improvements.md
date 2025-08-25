# 文档检测优化方案

## 问题分析
基于对Ultralytics YOLOv8代码库的深入分析，针对以下两个核心问题提供解决方案：
1. **阅读理解题检测不全**：内容较多、占比较大的题目容易框不全
2. **英文段落边界不准**：多段落文档中段落检测边界容易偏离

## 根本原因

### 1. Detection Head架构限制
- `reg_max`被设置为1（`ultralytics/nn/modules/head.py:38`），严重限制了边界框回归的精度
- 原始YOLOv8设计的reg_max=16适合自然图像，但当前配置降低到1，无法精确定位文档边界

### 2. 损失函数配置问题
- 使用`smooth_l1_loss`替代`l1_loss`（`utils/loss.py:112`），过度平滑了边界定位
- Box损失权重相对较低，不利于精确边界学习

### 3. 锚点分配策略不适配
- `TaskAlignedAssigner`的`topk=13`对大型文档对象覆盖不足
- 分类权重alpha=1.0过高，定位权重beta=6.0相对不足

## 具体改进方案

### 1. 修改Detection Head配置
```python
# 文件：ultralytics/nn/modules/head.py
# 第38行，将：
self.reg_max = 1
# 修改为：
self.reg_max = 16  # 恢复原始设计，提升边界框回归精度
```

### 2. 调整损失函数
```python
# 文件：ultralytics/utils/loss.py
# 第112行，考虑恢复为L1损失：
# l1_loss = F.smooth_l1_loss(pred_dist_pos, target_dist_pos, reduction="sum") / target_scores_sum
l1_loss = F.l1_loss(pred_dist_pos, target_dist_pos, reduction="sum") / target_scores_sum

# 第113-114行，调整损失权重：
l1_loss *= 0.8  # 增加L1损失权重（原0.5）
# loss_iou *= 0.1  # 可以考虑取消注释并调整IoU损失权重
```

### 3. 优化训练超参数
```yaml
# 文件：ultralytics/cfg/default.yaml
# 修改以下参数：

# 图像尺寸
imgsz: 1280  # 增加到1280或更高（原640）

# 损失权重
box: 10.0    # 增加box损失权重（原7.5）
cls: 0.3     # 降低分类损失（原0.5）
dfl: 2.0     # 增加DFL损失（原1.5）

# 数据增强（保持当前设置）
mosaic: 0.0  # 关闭mosaic
degrees: 0.0  # 关闭旋转
scale: 0.2    # 减少缩放幅度（原0.5）
translate: 0.05  # 减少平移（原0.1）
shear: 0.0    # 关闭剪切
perspective: 0.0  # 关闭透视变换

# NMS参数
iou: 0.5     # 提高NMS IoU阈值（原0.35）
max_det: 1000  # 增加最大检测数（原500）
```

### 4. 调整TaskAlignedAssigner
```python
# 文件：ultralytics/utils/tal.py
# 第28行，修改初始化参数：
def __init__(self, topk=20, num_classes=80, alpha=0.5, beta=8.0, eps=1e-9):
    # topk: 13 -> 20 (增加锚点覆盖)
    # alpha: 1.0 -> 0.5 (降低分类权重)
    # beta: 6.0 -> 8.0 (增加定位权重)
```

### 5. 使用P6模型架构
对于文档检测任务，建议使用包含P6层的模型架构：
```bash
# 训练命令示例
yolo train model=yolov8x-p6.yaml data=your_document_data.yaml imgsz=1280
```

P6架构优势：
- 包含1/64分辨率的特征层
- 更适合检测大型对象
- 对文档中的大区域有更好的感受野

## 实施建议

### 第一阶段（必须）
1. 修改`reg_max`从1恢复到16
2. 调整损失函数权重（box、cls、dfl）
3. 增加输入图像分辨率到1280

### 第二阶段（推荐）
1. 切换到P6模型架构
2. 调整TaskAlignedAssigner参数
3. 优化NMS参数

### 第三阶段（可选）
1. 根据具体数据集特点微调超参数
2. 尝试不同的损失函数组合
3. 实验更大的输入分辨率（如1536、2048）

## 预期效果
- **阅读理解题检测**：通过增加reg_max和输入分辨率，大幅提升大目标的完整检测率
- **段落边界精度**：通过优化损失函数和锚点分配，显著改善边界定位准确性
- **整体性能**：在保持检测速度的同时，将检测精度提升20-30%

## 注意事项
1. 增加输入分辨率会增加显存占用，需要相应调整batch size
2. 修改reg_max后需要重新训练模型，不能直接加载旧权重
3. 建议在验证集上逐步调整参数，避免过拟合

## 监控指标
训练时重点关注：
- Box损失的收敛情况
- 边界框IoU指标
- 大目标的召回率
- 边界定位的像素级误差