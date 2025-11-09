import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import cv2
from ultralytics import YOLO
from collections import defaultdict
import glob

# Load YOLO model
model = YOLO('yolo12x.pt')
class_list = model.names

# Path to image
cctv_frame = '/Users/elliott/Documents/GitHub/OpenSpot/image_gen/image_dataset/parking_rois_gopro/images/GOPR6775.JPG'

# Vehicle classes in COCO dataset (YOLO default)
vehicle_classes = {
    'car': 2,
    'motorcycle': 3,
    'bus': 5,
    'truck': 7
}

# Run detection
print(f"Running detection on: {cctv_frame}")
results = model(cctv_frame)

# Load image for visualization
img = cv2.imread(cctv_frame)

# Process detections
vehicle_detections = []
for result in results:
    boxes = result.boxes
    for box in boxes:
        # Get class ID and confidence
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = class_list[class_id]
        
        # Check if it's a vehicle
        if class_name in vehicle_classes.keys():
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Calculate center coordinates
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Store detection info
            detection_info = {
                'class': class_name,
                'confidence': confidence,
                'bbox': (x1, y1, x2, y2),
                'center': (center_x, center_y),
                'width': x2 - x1,
                'height': y2 - y1
            }
            vehicle_detections.append(detection_info)
            
            # Draw bounding box on image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw center point
            cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # Add label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 0), 2)
            
            # Print coordinates
            print(f"\n{class_name.upper()} detected:")
            print(f"  Confidence: {confidence:.2%}")
            print(f"  Bounding Box: Top-Left({x1}, {y1}), Bottom-Right({x2}, {y2})")
            print(f"  Center Point: ({center_x}, {center_y})")
            print(f"  Dimensions: {x2-x1}w x {y2-y1}h pixels")

# Summary
print(f"\n{'='*60}")
print(f"SUMMARY: {len(vehicle_detections)} vehicle(s) detected")
print(f"{'='*60}")

# Save annotated image
output_path = 'detected_vehicles.jpg'
cv2.imwrite(output_path, img)
print(f"\nAnnotated image saved to: {output_path}")

# Print all coordinates in a structured format
print("\n" + "="*60)
print("ALL VEHICLE COORDINATES:")
print("="*60)
for i, detection in enumerate(vehicle_detections, 1):
    print(f"{i}. {detection['class'].upper()}")
    print(f"   BBox: [{detection['bbox'][0]}, {detection['bbox'][1]}, "
          f"{detection['bbox'][2]}, {detection['bbox'][3]}]")
    print(f"   Center: ({detection['center'][0]}, {detection['center'][1]})")