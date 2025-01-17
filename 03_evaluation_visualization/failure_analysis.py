import cv2
from ultralytics import YOLO
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load YOLOv8 model
model = YOLO("/home/katomaran/Downloads/Boasch/Final/02_model_training/runs/detect/train2/weights/best.pt") 
CLASS_LABELS = [
    "person", "rider", "car", "truck", "bus", "train", "motor", "bike", 
    "traffic light", "traffic sign"
]

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

def compute_iou(box1, box2):
    """Compute IoU (Intersection over Union) between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0




def get_confusion_matrix(groundtruth, predictions, class_labels, iou_threshold=0.5):
    gt_labels = []
    pred_labels = []

    for gt in groundtruth:
        matched = False
        for pred in predictions:
            iou = compute_iou(gt['bbox'], pred['bbox'])
            if iou >= iou_threshold:
                gt_labels.append(gt['class_id'])
                pred_labels.append(pred['class_id'])
                matched = True
                break
        if not matched:
            gt_labels.append(gt['class_id'])  # Missed ground truth (FN)
            pred_labels.append(-1)  # No prediction for this ground truth

    for pred in predictions:
        if all(compute_iou(gt['bbox'], pred['bbox']) < iou_threshold for gt in groundtruth):
            gt_labels.append(-1)  # False positive prediction
            pred_labels.append(pred['class_id'])

    cm = confusion_matrix(gt_labels, pred_labels, labels=list(range(len(class_labels))) + [-1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels + ["FP/FN"])
    disp.plot(cmap="viridis", xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.show()

def visualize_predictions(image_path, txt_path, save_path=None):
    """Visualize YOLOv8 predictions and ground truth side by side."""
    image = cv2.imread(image_path)
    image_height, image_width = image.shape[:2]

    iou_threshold = 0.5

    ground_truth = load_ground_truth(txt_path, image_width, image_height)

    results = model(image)
    
    predictions = []
    for box in results[0].boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = box
        print(x1, y1, x2, y2)
        predictions.append({
            "class_id": int(cls),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
        })
    
    for gt in ground_truth:
        matched = False
        for pred in predictions:
            iou = compute_iou(gt['bbox'], pred['bbox'])
            if iou >= iou_threshold:
                all_true_labels.append(gt['class_id'])
                all_pred_labels.append(pred['class_id'])
                matched = True
                break
        if not matched:
            all_true_labels.append(gt['class_id'])  # Missed ground truth
            all_pred_labels.append(-1)  # No prediction for this ground truth

    for pred in predictions:
        if all(compute_iou(gt['bbox'], pred['bbox']) < iou_threshold for gt in ground_truth):
            all_true_labels.append(-1)  # False positive
            all_pred_labels.append(pred['class_id'])


    # Draw ground truth and predictions
    gt_image = draw_annotations(image.copy(), ground_truth, color=(0, 255, 0), label_prefix="GT: ")
    pred_image = draw_annotations(image.copy(), predictions, color=(255, 0, 0), label_prefix="PRED: ")

    # Combine images side by side
    combined_image = cv2.hconcat([gt_image, pred_image])

    # Display the combined image
    cv2.imshow("Ground Truth vs Predictions", combined_image)
    cv2.waitKey(0)
    get_confusion_matrix(ground_truth, predictions, CLASS_LABELS)



# Paths
image_folder = "/home/katomaran/Downloads/Boasch/Final/02_model_training/dataset/images/val/"
txt_folder = "/home/katomaran/Downloads/Boasch/Final/02_model_training/dataset/labels/val"
output_folder = "output/val/"

Path(output_folder).mkdir(exist_ok=True, parents=True)

all_true_labels = []
all_pred_labels = []
for image_file in Path(image_folder).glob("*.jpg"):
    txt_file = Path(txt_folder) / (image_file.stem + ".txt")
    if txt_file.exists():
        visualize_predictions(str(image_file), str(txt_file), save_path=f"{output_folder}/{image_file.name}")


cm = confusion_matrix(all_true_labels, all_pred_labels, labels=list(range(len(CLASS_LABELS))) + [-1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_LABELS + ["FP/FN"])
disp.plot(cmap="viridis", xticks_rotation=45)
plt.title("Confusion Matrix")
plt.show()