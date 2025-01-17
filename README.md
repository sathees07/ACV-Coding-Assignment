# Applied CV Coding Assignment

This repository contains the code, pipelines, and documentation for an end-to-end object detection project using the BDD100K dataset. The project encompasses:

- Dataset analysis
- Model training
- Model evaluation
- Containerized deployment for data tasks

Below is a breakdown of the repository structure and instructions for running and utilizing each task.

---

## 📂 Folder Structure

```
.
├── 01_data_analysis
│   ├── app.py
│   ├── config.json
│   ├── data-analysis.ipynb
│   ├── Dockerfile
│   ├── extract_od_data.py
│   ├── README.md
│   └── requirements.txt
├── 02_model_training
│   ├── bdd_to_yolo.py
│   ├── dataset
│   ├── dataset.yml
│   ├── runs
│   ├── training.ipynb
│   ├── training_steps.png
│   └── yolov8n.pt
├── 03_evaluation_visualization
│   ├── failure_analysis.py
│   ├── qual_performance_eval.py
│   └── quan_performance_eval.ipynb
└── README.md
```

# Data Analysis

The Data Analysis folder contains scripts to analyze and visualize the dataset using a Streamlit-based UI. The task involves filtering data, viewing labels, and performing exploratory data analysis (EDA).

Initial Setup
-------------

### 1\. Clone the GitHub Repository

To begin, clone the repository containing the analysis code:

git clone <https://github.com/sathees07/ACV-Coding-Assignment.git>

cd ACV-Coding-Assignment

### 2\. Download the Dataset

Download the BDD100K dataset from the provided link:

-   [Download Link](https://drive.google.com/file/d/1NgWX5YfEKbloAKX9l8kUVJFpWFlUO8UT/view?usp=sharing)

Place the downloaded files in the main folder.

### 3\. Data Preprocessing

The dataset contains annotations for object detection, drivable area, and lane markings in a JSON format. For this task, only object detection labels were extracted.

# Extraction Steps

1.  Navigate to the directory `01_data_analysis`

`cd 01_data_analysis`

1.  Use the provided parsing script to extract object detection data:\
    `python extract_od_data.py --input <path-to-json> --output <output-path>`

For example,

To extract from train dataset:\
`python3 extract_od_data.py --input ../assignment_data_bdd/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json --output od_train.json`

To extract from val dataset:\
`python3 extract_od_data.py --input ../assignment_data_bdd/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json --output od_val.json`

The script filters the JSON file to retain only the object detection annotations (e.g., car, person, traffic light, etc.) and stores the output in a simplified JSON format.


The folder structure is as follows:

```
.
├── app.py                  # Dashboard app for visualization
├── config.json             # Configuration file for app.py
├── data-analysis.ipynb     # Jupyter notebook used for developing app.py
├── Dockerfile              # Docker container file
├── extract_od_data.py      # Script for extracting object detection data
├── README.md
├── requirements.txt        # List of required packages
├── train.json              # Output file from data preprocessing (Train)
└── val.json                # Output file from data preprocessing (Val)

```

### Configuration File (config.json)

The config.json file provides the necessary configuration for running app.py. Its contents are as follows:

```
{

  "input_json": "od_val.json",

  "image_path": "assignment_data_bdd/bdd100k_images_100k/bdd100k/images/100k/val/"

}
```

#### Modifying the Configuration

-   To analyze the validation dataset, keep the default values:
```
input_json: od_val.json
image_path: assignment_data_bdd/bdd100k_images_100k/bdd100k/images/100k/val/

```

-   To analyze the training dataset, update the configuration as follows:

```
input_json: od_train.json
Image_path: assignment_data_bdd/bdd100k_images_100k/bdd100k/images/100k/train/
```

The analysis environment is fully containerized using Docker. The Dockerfile is located in this folder. To build and run the container:

```
docker build -t data-analysis-app .

docker run -p 8501:8501 -v $(pwd)/config.json:/app/config.json -v /<dataset_path>/assignment_data_bdd/:/app/assignment_data_bdd/ data-analysis-app .

```

- After running the command, Streamlit will automatically open the app in your default web browser.

- If the browser doesn't open automatically, check the terminal for the URL provided by Streamlit.

- Manually copy and paste the URL into your browser. Typically, this will be:

```
Local URL: http://localhost:8501
Network URL: http://0.0.0.0:8501
```

### Workflow

**1.  Select a Section:**

-   Use the sidebar to navigate to a specific analysis section.

**2.  Interact with Visualizations:**

-   View charts or interact with the dataset insights.

**3.  Explore Images:**

-   Use the Image Viewer to browse through images with annotations.

-   Use the Category-Based Viewer for focused analysis on a specific object category.


All the steps involved in the BDD100K dataset analysis app are documented in the ```01_data_analysis/README.md``` file. Please refer to that file for a comprehensive guide on how to:

-   Navigate through the UI.

-   Filter and analyze the BDD100K dataset.

-   Access specific functionalities like category filtering and label-based image exploration.
