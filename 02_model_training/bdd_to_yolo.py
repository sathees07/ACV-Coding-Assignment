import json
from PIL import Image
import os

# Paths
train_path = "../../assignment_data_bdd/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json"
val_path = "../../assignment_data_bdd/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json"

# Input Image Path
train_images_folder = "dataset/images/train"
val_images_folder = "dataset/images/val"

# Output Label path
train_output_dir = "dataset/labels/train"
val_output_dir = "dataset/labels/val"

# Ensure the output directories exist
os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(val_output_dir, exist_ok=True)

# Mapping of labels to IDs
label2id = {
    "person": 0,
    "rider": 1,
    "car": 2,
    "truck": 3,
    "bus": 4,
    "train": 5,
    "motor": 6,
    "bike": 7,
    "traffic light": 8,
    "traffic sign": 9
}

def convert_to_yolo_format(x1, y1, x2, y2, img_width, img_height, class_id=0):
    """Convert bounding box to YOLO format."""
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    width = x2 - x1
    height = y2 - y1

    norm_center_x = center_x / img_width
    norm_center_y = center_y / img_height
    norm_width = width / img_width
    norm_height = height / img_height

    return f"{class_id} {norm_center_x:.6f} {norm_center_y:.6f} {norm_width:.6f} {norm_height:.6f}"

def process_annotations(json_path, image_folder, output_folder):
    """Process annotations and convert them to YOLO format."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        for annotation in data:
            image_path = os.path.join(image_folder, annotation['name'])

            if not os.path.exists(image_path):
                print(f"Warning: Image not found {image_path}, skipping.")
                continue

            print(f"Processing: {image_path}")
            
            try:
                img = Image.open(image_path)
                width, height = img.size
                text = ''

                for label in annotation['labels']:
                    if 'box2d' not in label:
                        continue

                    classid = label2id.get(label['category'])
                    if classid is None:
                        print(f"Warning: Unknown category {label['category']}, skipping.")
                        continue

                    x1, y1, x2, y2 = label['box2d']['x1'], label['box2d']['y1'], label['box2d']['x2'], label['box2d']['y2']
                    text += convert_to_yolo_format(x1, y1, x2, y2, width, height, classid) + '\n'

                # Save annotations to a text file
                output_file = os.path.join(output_folder, annotation['name'].replace('jpg', 'txt'))
                with open(output_file, 'w') as f:
                    f.write(text)

            except Exception as e:
                print(f"Error processing image {image_path}: {e}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON file {json_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":

    print("Processing training dataset...")
    process_annotations(train_path, train_images_folder, train_output_dir)

    print("Processing validation dataset...")
    process_annotations(val_path, val_images_folder, val_output_dir)

    print("Processing complete.")
