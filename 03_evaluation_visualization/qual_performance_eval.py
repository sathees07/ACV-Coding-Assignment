import cv2
from ultralytics import YOLO
from pathlib import Path


# Paths
image_folder = "02_model_training/dataset/images/train/"
txt_folder = "02_model_training/dataset/labels/train"
output_folder = "output/train/"
model_path = "02_model_training/runs/detect/train2/weights/best.pt"

CLASS_LABELS = [
    "person", "rider", "car", "truck", "bus", "train", "motor", "bike", 
    "traffic light", "traffic sign"
]

# Load YOLOv8 model
model = YOLO(model_path) 

def load_ground_truth(txt_file, image_width, image_height):
    """Load ground truth annotations from a YOLO TXT file."""
    with open(txt_file, 'r') as f:
        annotations = []
        for line in f:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            x1 = int((x_center - width / 2) * image_width)
            y1 = int((y_center - height / 2) * image_height)
            x2 = int((x_center + width / 2) * image_width)
            y2 = int((y_center + height / 2) * image_height)
            annotations.append({"class_id": int(class_id), "bbox": [x1, y1, x2, y2]})
    return annotations

def draw_annotations(image, annotations, color=(0, 255, 0), label_prefix=""):
    """Draw bounding boxes and labels on an image."""
    for ann in annotations:
        x1, y1, x2, y2 = ann['bbox']
        label = f"{label_prefix}{CLASS_LABELS[ann['class_id']]}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

def visualize_predictions(image_path, txt_path, save_path=None):
    """Visualize YOLOv8 predictions and ground truth side by side."""
    # Load image
    image = cv2.imread(image_path)
    image_height, image_width = image.shape[:2]

    # Load ground truth
    ground_truth = load_ground_truth(txt_path, image_width, image_height)

    # Run YOLOv8 model for predictions
    results = model(image)

    # Parse predictions
    predictions = []
    for box in results[0].boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = box
        print(x1, y1, x2, y2, conf)
        predictions.append({
            "class_id": int(cls),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
        })

    # Draw ground truth and predictions
    gt_image = draw_annotations(image.copy(), ground_truth, color=(0, 255, 0), label_prefix="GT: ")
    pred_image = draw_annotations(image.copy(), predictions, color=(255, 0, 0), label_prefix="PRED: ")

    # Combine images side by side
    combined_image = cv2.hconcat([gt_image, pred_image])

    # Display the combined image
    cv2.imshow("Ground Truth vs Predictions", combined_image)
    cv2.waitKey(0)

    # Save the image if a save path is provided
    if save_path:
        cv2.imwrite(save_path, combined_image)



Path(output_folder).mkdir(exist_ok=True, parents=True)

# Process each image and its corresponding ground truth
for image_file in Path(image_folder).glob("*.jpg"):
    txt_file = Path(txt_folder) / (image_file.stem + ".txt")
    if txt_file.exists():
        visualize_predictions(str(image_file), str(txt_file), save_path=f"{output_folder}/{image_file.name}")
