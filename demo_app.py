import os
import sys
import importlib
from projects import ProjectBase
sys.path.append(os.getcwd())
from flask import Flask, render_template, request
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
from rotated_rect_utils import points4_to_lefttop_rightbottom_theta, rotated_rect_contains, crop_image_by_region_with_rotation

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()

# Define available models
AVAILABLE_MODELS = [
    '/mnt/ceph2/kuaiduya-engine/models/detection/line.pt',
    '/mnt/ceph2/kuaiduya-engine/models/detection/word.pt',
    # Add more model paths here as needed
]

# Load model (you can change this to your trained model)
model = None

def draw_box(image, _4pts_box, color, width=2):
    """Draw rotated bounding box on image"""
    draw = ImageDraw.Draw(image)
    x1, y1 = _4pts_box[0]
    x2, y2 = _4pts_box[1]
    x3, y3 = _4pts_box[2]
    x4, y4 = _4pts_box[3]
    draw.line([(x1, y1), (x2, y2)], width=width, fill=color)
    draw.line([(x2, y2), (x3, y3)], width=width, fill=color)
    draw.line([(x3, y3), (x4, y4)], width=width, fill=color)
    draw.line([(x4, y4), (x1, y1)], width=width, fill=color)

def draw_text(image, text, text_position, color, font_size=16):
    """Draw text on image"""
    draw = ImageDraw.Draw(image)
    try:
        # Try to use custom font if available
        font = ImageFont.truetype(font='/mnt/data5/kuaiduya/configs/fonts/msyh.ttc', size=font_size)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
    draw.text(text_position, text, fill=color, font=font)


loaded_models = {}

def process_image(image, model_path=None, conf_threshold=0.25, iou_threshold=0.45, project:ProjectBase=None, selected_classes:list[str]|None=None):
    image = image.copy()
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
    
    """Process image with YOLO model and draw detections"""
    global model
    
    if model_path not in loaded_models:
        loaded_models[model_path] = YOLO(model_path)
    model = loaded_models[model_path]
    image_ori = image
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # Always expand image
    image_expand = Image.new('RGB', (image.width * 2, image.height * 2), (255, 255, 255))
    image_expand.paste(image, (0, 0))
    image = image_expand
    # Run inference
    result = model.predict(image, imgsz=project.image_size(), conf=conf_threshold, iou=iou_threshold, max_det=1000, verbose=False)[0]
    image = image_ori
    # Create a copy for drawing
    names = result.names
    obb = result.obb
    cls = obb.cls.tolist()
    conf = obb.conf.tolist()
    boxes = obb.xyxyxyxy.tolist()
    predicted = []
    for i in range(len(cls)):
        _4pts_box = boxes[i]
        # Convert tuples to lists so we can modify them
        _4pts_box = [list(pt) for pt in _4pts_box]
        for pt in _4pts_box:
            pt[0] = min(max(0, pt[0]), image.width)
            pt[1] = min(max(0, pt[1]), image.height)
        box = points4_to_lefttop_rightbottom_theta(_4pts_box)
        
        cls_1 = project.raw_to_cls(int(cls[i]))
        assert cls_1 is not None, f'mapper error, {cls[i]}'
        color = project.cls_to_color(cls_1)
        if selected_classes is not None and project.cls_to_name(cls_1) not in selected_classes:
            continue
        croped = crop_image_by_region_with_rotation(img_cv, list(box[:4]), box[4])
        croped = Image.fromarray(cv2.cvtColor(croped, cv2.COLOR_BGR2RGB))

        predicted.append({
            'cls' : cls_1,
            'region' : list(box[:4]),
            'rotation' : box[4],
            'box' : box,
            '4pts_box' : _4pts_box,
            'score' : conf[i],
            'color' : (color[2], color[1], color[0]),
            'croped' : croped
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

    for predicted_item in predicted:
        #predicted_item['4pts_box']
        _4pts_box = predicted_item['4pts_box']
        color = predicted_item['color']
        
        score = predicted_item['score']
        draw_box(image, _4pts_box, color)
        if score < 0.25:
            draw = ImageDraw.Draw(image)
            font = ImageFont.load_default()
            text = f"{score}"
            text_position = (_4pts_box[0][0], _4pts_box[0][1])
            draw.text(text_position, text, fill=color, font=font)
    
    return image, predicted

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/get_projects')
def get_projects():
    """Get list of available projects"""
    try:
        projects_dir = os.path.join(os.getcwd(), 'projects')
        projects = []
        
        # List all directories in projects folder
        for item in os.listdir(projects_dir):
            item_path = os.path.join(projects_dir, item)
            # Check if it's a directory and has common.py file
            if os.path.isdir(item_path) and item != '__pycache__':
                common_file = os.path.join(item_path, 'common.py')
                if os.path.exists(common_file):
                    projects.append({
                        'value': item,
                        'name': item.replace('_', ' ').title()
                    })
        
        return {'success': True, 'projects': projects}
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/get_models')
def get_models():
    """Get list of available models"""
    try:
        # Check which models actually exist
        available_models = []
        for model_path in AVAILABLE_MODELS:
            if os.path.exists(model_path):
                available_models.append(model_path)
        
        return {'success': True, 'models': available_models}
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/get_classes/<project_name>')
def get_classes(project_name):
    """Get available classes for a project"""
    try:
        project = importlib.import_module(f'projects.{project_name}.common').Project()
        classes = project.all_cls()
        classes = [project.cls_to_name(cls) for cls in classes]
        return {'success': True, 'classes': classes}
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/detect', methods=['POST'])
def detect():
    """Handle image upload and detection"""
    if 'file' not in request.files:
        return {'error': 'No file uploaded'}, 400
    
    file = request.files['file']
    if file.filename == '':
        return {'error': 'No file selected'}, 400
    
    # Get parameters
    conf_threshold = float(request.form.get('conf', 0.25))
    iou_threshold = float(request.form.get('iou', 0.45))
    model_path = request.form.get('model_path', '')
    project = request.form.get('project', '')
    project = importlib.import_module(f'projects.{project}.common').Project()
    selected_classes = request.form.get('selected_classes', '')
    selected_classes = selected_classes.split(',') if selected_classes else None
    
    try:
        # Read and process image
        image = Image.open(file.stream).convert('RGB')
        
        # Process image with expand (always enabled)
        result_image, detections = process_image(
            image, 
            model_path=model_path if model_path else None,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            project = project,
            selected_classes=selected_classes,
        )
        
        # Convert result image to base64
        buffered = io.BytesIO()
        result_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Convert cropped images to base64
        for detection in detections:
            detection['cls'] = project.cls_to_name(detection['cls'])
            if 'croped' in detection:
                buffered_crop = io.BytesIO()
                detection['croped'].save(buffered_crop, format="PNG")
                detection['croped_base64'] = base64.b64encode(buffered_crop.getvalue()).decode('utf-8')
                del detection['croped']  # Remove PIL Image object
        
        return {
            'success': True,
            'result_image': f"data:image/png;base64,{img_base64}",
            'detections': detections,
            'num_detections': len(detections)
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'error': str(e)}, 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)