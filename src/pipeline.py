import cv2
from PIL import Image
import matplotlib.pyplot as plt
from src.yolo_detector import YOLODetector
from src.blip_captioner import BLIPCaptioner

def main():
    # Initialize models
    detector = YOLODetector("models/yolov8m-worldv2.onnx", 
                           class_prompts=["a red triangular traffic sign", "a stop sign"],
                           device="cpu")
                           
    captioner = BLIPCaptioner()
    
    # Load image and change it to RGB
    image_path = "data/foggy1.png"
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Detect objects
    boxes, scores, class_ids = detector.detect(image_rgb)
    
    # Process each detection
    for box, score, class_id in zip(boxes, scores, class_ids):
        # Crop object
        x, y, w, h = box
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)
        
        crop = image_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue
            
        # Generate caption
        caption = captioner.generate_caption(Image.fromarray(crop))
        print(f"Detected: {caption}")
        
        # Draw results
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_bgr, caption, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Display results
    plt.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
