import math
import cv2, imutils
import numpy as np
from PIL import Image
from copy import deepcopy


# def three_points_to_center_wh_theta(three_points, to_int):
#    x1, y1, x2, y2, x3, y3 = three_points
#    x_c = (x1 + x3) / 2
#    y_c = (y1 + y3) / 2
#
#    if to_int:
#        x_c = int(x_c)
#        y_c = int(y_c)
#
#    w = math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))
#    h = math.sqrt(math.pow(x2 - x3, 2) + math.pow(y2 - y3, 2))
#
#    arc_tan =


def three_points_to_lefttop_wh_theta(three_points, to_int):
    x1, y1, x2, y2, x3, y3 = three_points
    w = math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))
    h = math.sqrt(math.pow(x2 - x3, 2) + math.pow(y2 - y3, 2))
    theta = math.degrees(math.atan2(y2 - y1, x2 - x1))
    ret = (x1, y1, w, h, theta)
    if not to_int:
        return ret
    else:
        return map(lambda x: int(round(x)), ret)


def three_points_to_lefttop_rightbottom_theta(three_points, to_int):
    x1, y1, w, h, theta = three_points_to_lefttop_wh_theta(three_points, False)

    x2 = x1 + w
    y2 = y1 + h

    ret = (x1, y1, x2, y2, theta)
    if not to_int:
        return ret
    else:
        return map(lambda x: int(round(x)), ret)

# def topleft_wh_theta_to_three_points(region, to_int):
#    x1, y1, w, h, theta = region


def three_points_crop_img(img, three_points):
    region = three_points_to_lefttop_rightbottom_theta(three_points, True)
    return lefttop_rightbottom_theta_crop_img(img, region)


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

def points4_to_lefttop_rightbottom_theta(four_points):
    
    """
    Reverse function: converts 4 rotated points back to the original bounding box format.
    
    Args:
        rotated_points: List of 4 points from lefttop_rightbottom_theta_to_4points
    
    Returns:
        tuple: (x1, y1, x2, y2, theta) - the original bounding box coordinates and rotation
    """
    points = []
    for p in four_points:
        if len(p) == 3:
            points.append((p[0], p[1]))
        else:
            points.append(p)
    
    # 找到左上角点（x+y最小）
    min_idx = 0
    min_sum = points[0][0] + points[0][1]
    for i in range(1, 4):
        s = points[i][0] + points[i][1]
        if s < min_sum:
            min_sum = s
            min_idx = i
    # 旋转点的顺序，使左上角为第一个
    points = points[min_idx:] + points[:min_idx]

    dx = points[1][0] - points[0][0]
    dy = points[1][1] - points[0][1]
    theta = np.degrees(np.arctan2(dy, dx))  

    #如果第一条边是竖着的，则倒转4个点
    if theta > 45:
        points = [points[0], points[3], points[2], points[1]]
    
    x1, y1 = points[0]
    
    dx = points[1][0] - points[0][0]
    dy = points[1][1] - points[0][1]
    theta = np.degrees(np.arctan2(dy, dx))  
    
    width = math.sqrt(dx*dx + dy*dy)
    
    dx2 = points[2][0] - points[1][0]
    dy2 = points[2][1] - points[1][1]
    height = math.sqrt(dx2*dx2 + dy2*dy2)
    
    x2 = x1 + width
    y2 = y1 + height
    
    x1 = round(x1, 10)
    y1 = round(y1, 10)
    x2 = round(x2, 10)
    y2 = round(y2, 10)
    theta = round(theta, 10)
    return (x1, y1, x2, y2, theta)




def lefttop_rightbottom_theta_to_center_wh_theta(topleft):
    x_min, y_min, x_max, y_max, theta = topleft
    w = x_max - x_min
    h = y_max - y_min

    _4points = lefttop_rightbottom_theta_to_4points(topleft)
    p1 = _4points[0]
    p3 = _4points[2]
    x_center = (p1[0] + p3[0]) / 2
    y_center = (p1[1] + p3[1]) / 2

    return (x_center, y_center, w, h, theta)


def draw_lefttop_rightbottom_theta(img, region, color, width = 1):
    _4points = lefttop_rightbottom_theta_to_4points(region)
    x1, y1 = map(lambda x: int(x), _4points[0])
    x2, y2 = map(lambda x: int(x), _4points[1])
    x3, y3 = map(lambda x: int(x), _4points[2])
    x4, y4 = map(lambda x: int(x), _4points[3])
    cv2.line(img, (x1, y1), (x2, y2), color, width)
    cv2.line(img, (x2, y2), (x3, y3), color, width)
    cv2.line(img, (x3, y3), (x4, y4), color, width)
    cv2.line(img, (x4, y4), (x1, y1), color, width)

def rotated_rect_contains_ratio(big_box, small_box):
    '''
    big_box: format lefttop_rightbottom_theta
    small_box: format lefttop_rightbottom_theta
    '''
    big_one = lefttop_rightbottom_theta_to_center_wh_theta(big_box)
    small_one = lefttop_rightbottom_theta_to_center_wh_theta(small_box)

    cv_big = ((big_one[0], big_one[1]), (big_one[2], big_one[3]), big_one[4])
    cv_small = ((small_one[0], small_one[1]), (small_one[2], small_one[3]), small_one[4])

    int_pts = cv2.rotatedRectangleIntersection(cv_big, cv_small)[1]
    if int_pts is None:
        return 0

    small_area = cv_small[1][0] * cv_small[1][1]
    order_pts = cv2.convexHull(int_pts, returnPoints=True)
    int_area = cv2.contourArea(order_pts)
    return int_area / small_area 

def rotated_rect_contains(big_box, small_box, thresh=0.8):
    
    return rotated_rect_contains_ratio(big_box, small_box) >= thresh


def rotated_rect_iou(box1, box2):
    '''
    计算两个旋转矩形的IoU
    box1, box2: 格式 (left, top, right, bottom, theta)
    '''
    # 转换为 OpenCV 格式 ((cx, cy), (w, h), theta)
    rect1 = lefttop_rightbottom_theta_to_center_wh_theta(box1)
    rect2 = lefttop_rightbottom_theta_to_center_wh_theta(box2)

    cv_rect1 = ((rect1[0], rect1[1]), (rect1[2], rect1[3]), rect1[4])
    cv_rect2 = ((rect2[0], rect2[1]), (rect2[2], rect2[3]), rect2[4])

    # 计算交集区域
    int_pts = cv2.rotatedRectangleIntersection(cv_rect1, cv_rect2)[1]
    if int_pts is None:  # 没有交集
        return 0.0

    # 计算交集面积
    int_area = cv2.contourArea(cv2.convexHull(int_pts))

    # 计算每个矩形的面积
    area1 = rect1[2] * rect1[3]
    area2 = rect2[2] * rect2[3]

    # 计算IoU
    iou = int_area / (area1 + area2 - int_area)
    return iou


def lefttop_reightbottom_theta_bound_box(region, to_int):
    _4points = lefttop_rightbottom_theta_to_4points(region)
    x_min = min([x[0] for x in _4points])
    y_min = min([x[1] for x in _4points])
    x_max = max([x[0] for x in _4points])
    y_max = max([x[1] for x in _4points])
    ret = [x_min, y_min, x_max, y_max]
    if not to_int:
        return ret
    else:
        return [int(round(x)) for x in ret]


def lefttop_rightbottom_theta_crop_img(img, region):
    x1, y1, x2, y2, theta = region

    if theta == 0:
        rotated_img = img
        return rotated_img[y1:y2, x1:x2]
    else:
        x_min, y_min, x_max, y_max = lefttop_reightbottom_theta_bound_box(region, False)
        # simple fix bug x_min, y_min become < 0
        x_min = max(0, int(math.floor(x_min)))
        y_min = max(0, int(math.floor(y_min)))
        x_max = int(math.ceil(x_max))
        y_max = int(math.ceil(y_max))

        crop_img = img[y_min:y_max, x_min:x_max]
        ih, iw = crop_img.shape[:2]
        M = cv2.getRotationMatrix2D((x1 - x_min, y1 - y_min), theta, 1)
        rotated_img = cv2.warpAffine(crop_img, M, (max(iw, x2 - x_min),
                                                   max(ih,  y2 - y_min)), borderValue=(255, 255, 255))

        return rotated_img[y1 - y_min:y2 - y_min, x1 - x_min:x2 - x_min]


FIXED_IMAGE_SIZE = 224


def crop_and_rotate_for_search(img, region_with_rotation=None, debug_file=None):
    crop = img
    if region_with_rotation:
        crop = lefttop_rightbottom_theta_crop_img(img, region_with_rotation)
    crop = cv2.resize(crop, (FIXED_IMAGE_SIZE, FIXED_IMAGE_SIZE))
    if debug_file:
        cv2.imwrite(debug_file, crop)
    return crop


def imutils_resize(image, width=None, height=None, inter=cv2.INTER_LINEAR):
    img2 = imutils.resize(image, width=width, height= height, inter=inter)
    return img2


def resize_short_side(img, _side):
    h, w = img.shape[:2]
        
    img2 = img
    if min(h, w) > _side:
        if h > w:
            img2 = imutils_resize(img, width=int(_side))
        else:
            img2 = imutils_resize(img, height=int(_side))
    
    return img2

def makesure_point_in_wh(x1,y1, iw, ih):
#     ih, iw = img.shape[:2]
    x1 = max(x1, 1)
    y1 = max(y1, 1)
    x1 = int(min(x1, iw-1))
    y1 = int(min(y1, ih-1))
    
    return x1,y1

def makesure_3pointbox_in_wh(three_points, iw, ih):
    x1, y1, x2, y2, x3, y3 = three_points
#     ih, iw = img.shape[:2]
    x1,y1 = makesure_point_in_wh(x1,y1,iw, ih)
    x2,y2 = makesure_point_in_wh(x2,y2,iw, ih)
    x3,y3 = makesure_point_in_wh(x3,y3,iw, ih)
    
    three_points2 = (x1, y1, x2, y2, x3, y3)
    
    return three_points2


def crop_image_by_region_with_rotation(image:Image.Image|cv2.typing.MatLike, region:list, rotation:int|float)->Image.Image:
    if isinstance(image, Image.Image):
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        img_cv = image
    h_center = (region[1] + region[3]) // 2
    height = region[3] - region[1]
    #height = int(height * 0.85)
    region1 = deepcopy(region)
    region1[1] = h_center - height // 2
    region1[3] = h_center + height // 2
    #print(region['region'], region1)
    pts = lefttop_rightbottom_theta_to_4points(region1 + [rotation])
    # 将四个点转换为numpy数组
    pts = np.float32(pts)
    # 计算目标矩形的宽度和高度
    width = int(np.linalg.norm(pts[1] - pts[0]))
    height = int(np.linalg.norm(pts[3] - pts[0]))
    # 定义目标矩形的四个角点
    dst_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(pts, dst_pts)
    # 将PIL Image转换为OpenCV格式
    
    # 进行透视变换
    warped = cv2.warpPerspective(img_cv, M, (width, height))
    # 转回PIL Image格式
    if isinstance(image, Image.Image):
        warped_pil = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    else:
        warped_pil = warped
    return warped_pil


if __name__ =='__main__':
    region = [100, 200, 500, 300, 10]
    _4points = lefttop_rightbottom_theta_to_4points(region)
    print("4 points:", _4points)
    region1 = points4_to_lefttop_rightbottom_theta(_4points)
    print("Reconstructed region:", region1)
    
    # Compare with tolerance for floating point errors
    tolerance = 1e-9
    assert abs(region1[0] - region[0]) < tolerance, f"x1 mismatch: {region1[0]} != {region[0]}"
    assert abs(region1[1] - region[1]) < tolerance, f"y1 mismatch: {region1[1]} != {region[1]}"
    assert abs(region1[2] - region[2]) < tolerance, f"x2 mismatch: {region1[2]} != {region[2]}"
    assert abs(region1[3] - region[3]) < tolerance, f"y2 mismatch: {region1[3]} != {region[3]}"
    assert abs(region1[4] - region[4]) < tolerance, f"theta mismatch: {region1[4]} != {region[4]}"
    print("Test passed!")
