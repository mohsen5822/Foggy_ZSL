import cv2
from yolo_world_onnx import YOLOWORLD

class YOLODetector:
    def __init__(self, model_path, class_prompts, device="cpu"):
        self.model = YOLOWORLD(model_path, device=device)
        self.set_classes(class_prompts)
        
    def set_classes(self, class_prompts):
        self.model.set_classes(class_prompts)
        
    def detect(self, image, conf=0.1, imgsz=640, iou=0.1):
        """Detects objects in an image."""
        boxes, scores, class_ids = self.model(image, conf=conf, imgsz=imgsz, iou=iou)
        return boxes, scores, class_ids
        
    def draw_detections(self, image, boxes, scores, class_ids, class_names):
        """Draws detections on the image."""
        # ... (your drawing code here)
        return image