import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import json
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import os
from itertools import combinations



# Streamlit app title
st.title('BDD100K Dataset Analysis')

with open("config.json", "r") as f:
    config = json.load(f)

# Hardcoded dataset loading
with open(config["input_json"], "r") as f:
    data = json.load(f)

image_path = config["image_path"]
categories = ["car", "traffic light", "traffic sign", "bike","truck","person","bus","train","motor","rider"]


# Sidebar navigation
section = st.sidebar.radio(
    "Select Analysis Section",
    ("Dataset Overview", "Object Category Distribution", "Weather Conditions", "Time of Day", "Scene Distribution", "Bounding Box Statistics",
     "Object Density Heatmap Analysis","Advanced Spatial Heatmap Analysis",
     "Category Correlation Analysis","Annotations Quality Analysis","Object Co-Occurrence Analysis",
     "Category-Based Viewer")
)

# Dataset Overview
if section == "Dataset Overview":
    st.header("Dataset Overview")
    st.write(f"Number of images in the dataset: {len(data)}")
    if st.checkbox("Show first record"):
        st.json(data[0])

    
    st.header("Image Viewer")
    st.write(f"Number of images in the dataset: {len(data)}")


    if 'image_index' not in st.session_state:
        st.session_state.image_index = 0

    def load_image(index):
        item = data[index]
        img_name = item["name"]
        img_path = os.path.join(image_path, img_name)
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            st.error(f"Image not found: {img_path}")
            return None
        draw = ImageDraw.Draw(img)
        for label in item["labels"]:
            if "box2d" in label:
                box = label["box2d"]
                draw.rectangle([box["x1"], box["y1"], box["x2"], box["y2"]], outline="red", width=2)
                draw.text((box["x1"], box["y1"] - 10), label["category"], fill="white")
        return img, item["attributes"]

    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        if st.button("Prev"):
            st.session_state.image_index = max(0, st.session_state.image_index - 1)
    with col3:
        if st.button("Next"):
            st.session_state.image_index = min(len(data) - 1, st.session_state.image_index + 1)

    img, attributes = load_image(st.session_state.image_index)
    if img:
        st.image(img, caption=f"Image {st.session_state.image_index + 1} of {len(data)}", use_container_width=True)
        st.write(f"**Weather:** {attributes['weather']} | **Scene:** {attributes['scene']} | **Time of Day:** {attributes['timeofday']}")


# Object Category Distribution
elif section == "Object Category Distribution":
    st.header("Object Category Distribution")
    st.write(f"Object Category Distribution refers to the frequency and proportion of different object classes present within a dataset. In object detection datasets like BDD100K, it shows how often each category (e.g., cars, pedestrians, traffic lights) appears. Analyzing this distribution helps identify class imbalances, guiding data preprocessing, model training, and performance evaluation to ensure accurate detection across all categories.")
    category_counts = Counter()
    for item in data:
        for label in item["labels"]:
            category_counts[label["category"]] += 1

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=list(category_counts.keys()), y=list(category_counts.values()), ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_title("Object Category Distribution")
    ax.set_ylabel("Count")
    ax.set_xlabel("Category")

    # Add count labels on bars
    for i, v in enumerate(category_counts.values()):
        ax.text(i, v + 0.5, str(v), ha='center', va='bottom')

    st.pyplot(fig)

    st.subheader("Anomaly Insights:")
    st.write("**Dominance of the Car Category → Anomalously High**")
    st.write("The **car** category vastly outnumbers all other classes, creating a significant class imbalance.")
    st.write("This may cause the detection model to overfit on cars while underperforming on minority classes.")

    st.write("**Severely Underrepresented Classes:**")
    st.write("**Train** : Extremely Low")
    st.write("**Motor, Rider, and Bike** also have very low counts compared to dominant classes.")
    st.write("These low frequencies may lead to poor detection accuracy for these objects.")

# Weather Conditions Distribution
elif section == "Weather Conditions":
    st.header("Weather Conditions Distribution")
    st.write("Weather Conditions Distribution represents how various weather scenarios (e.g., clear, rainy, snowy, foggy) are spread across a dataset. In datasets like BDD100K, this distribution helps analyze how balanced the data is across different weather conditions, ensuring the object detection model can perform well in diverse environments. A balanced distribution improves model robustness in real-world applications, while an imbalance might require data augmentation or re-sampling for optimal performance.")
    weather_counts = Counter()
    for item in data:
        weather_counts[item["attributes"]["weather"]] += 1

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(weather_counts.values(), labels=weather_counts.keys(), autopct="%1.1f%%")
    ax.set_title("Weather Conditions Distribution")
    st.pyplot(fig)

    st.subheader("Anomaly Insights:")
    st.write("**Foggy** → *Anomalously Low*\n\nThis count is extremely low compared to others, suggesting either rare occurrences or data collection bias.")
    st.write("**Clear** → *Anomalously High*\n\nThis count heavily dominates the dataset, indicating possible class imbalance. It could skew the model to perform better under clear conditions but poorly under others.")
    st.write("**Balanced Categories:**\n\nOvercast, Undefined, Snowy, Rainy, and Partly Cloudy appear more balanced relative to each other.")


# Time of Day Distribution
elif section == "Time of Day":
    st.header("Time of Day Distribution")
    st.write("Time of Day Distribution refers to how data samples are spread across different times of the day, such as daytime, nighttime, dawn, and dusk. In datasets like BDD100K, this distribution is crucial for training models to perform well under varying lighting conditions. Analyzing this helps identify imbalances (e.g., more daytime images than nighttime), guiding data augmentation and model tuning for better performance in real-world scenarios.")
    timeofday_counts = Counter()
    for item in data:
        timeofday_counts[item["attributes"]["timeofday"]] += 1

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=list(timeofday_counts.keys()), y=list(timeofday_counts.values()), ax=ax)
    ax.set_title("Time of Day Distribution")
    ax.set_ylabel("Count")
    ax.set_xlabel("Time of Day")
    for i, v in enumerate(timeofday_counts.values()):
        ax.text(i, v + 0.5, str(v), ha='center', va='bottom')
    st.pyplot(fig)

    st.subheader("Anomaly Insights:")
    st.write("**Daytime** → *Dominantly High*\n\nThis category is the most frequent, which may cause the model to perform better in well-lit conditions but struggle in low-light scenarios.")
    st.write("**Dawn/Dusk** → *Underrepresented*\n\nThis count is significantly lower than Daytime and Night, indicating potential bias in transitional lighting conditions (e.g., sunrise or sunset).")
    st.write("**Undefined** → *Anomalously Low*\n\nExtremely low and potentially problematic. This could result from labeling errors or missing data, which may need correction or exclusion.")
    st.write("**Night** → *Moderately Balanced*\n\nThough lower than Daytime, the Night category has a reasonable presence, supporting model generalization in darker conditions.")


# Scene Distribution
elif section == "Scene Distribution":
    st.header("Scene Distribution")
    st.write("Scene Distribution refers to how data samples are spread across different environmental settings or locations, such as city streets, highways, residential areas, rural roads, and parking lots. In datasets like BDD100K, analyzing scene distribution helps ensure that object detection models are trained on diverse environments, improving their ability to generalize across various real-world scenarios. Detecting imbalances in scene distribution is essential for balanced model performance across all scene types.")
    scene_counts = Counter()
    for item in data:
        scene_counts[item["attributes"]["scene"]] += 1

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=list(scene_counts.keys()), y=list(scene_counts.values()), ax=ax)
    ax.set_title("Scene Distribution")
    ax.set_ylabel("Count")
    ax.set_xlabel("Scene Type")
    for i, v in enumerate(scene_counts.values()):
        ax.text(i, v + 0.5, str(v), ha='center', va='bottom')
    st.pyplot(fig)

    st.subheader("Anomaly Insights:")
    st.write("**Anomalously High:** → *City Street*\n\nThis category heavily dominates the dataset, indicating a strong focus on urban environments. Such dominance could bias the model to perform well in urban areas while struggling in other scenes like highways or rural settings.")
    st.write("**Balanced Categories:** → *Highway and Residential*\n\nThese categories are relatively well-represented compared to others and fall within the normal statistical range. They provide sufficient variation to train the model in less urbanized settings.")
    st.write("**Anomalously Low** → *Gas Stations and Tunnel*\n\nThese categories have extremely low counts, suggesting either rare occurrences in the dataset or potential data collection bias. Their limited representation could lead to poor detection performance in these scenes.")
    st.write("*Parking Lot and Undefined*\n\nThese are also underrepresented but not as extreme. They could still affect the model's ability to generalize effectively in these environments.")



# Bounding Box Statistics
elif section == "Bounding Box Statistics":
    st.header("Bounding Box Statistics")
    st.write("Bounding Box Statistics provide insights into the properties of annotated bounding boxes in an object detection dataset.Analyzing bounding box statistics helps identify patterns or anomalies, such as extremely small or large boxes, skewed aspect ratios, or uneven object distribution, which could affect model performance. Understanding these statistics is essential for preprocessing and optimizing the dataset for training robust object detection models.")

    bbox_data = []
    for item in data:
        for label in item["labels"]:
            if "box2d" in label:
                bbox = label["box2d"]
                width = bbox["x2"] - bbox["x1"]
                height = bbox["y2"] - bbox["y1"]
                bbox_data.append({"category": label["category"], "width": width, "height": height})

    # Convert to DataFrame
    bbox_df = pd.DataFrame(bbox_data)

    # Plot bounding box dimensions
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(bbox_df["width"], kde=True, bins=30, label="Width")
    sns.histplot(bbox_df["height"], kde=True, bins=30, label="Height", color="orange")
    ax.set_title("Bounding Box Dimensions")
    ax.set_xlabel("Dimension (pixels)")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)
    st.subheader("Anomaly Insights:")
    st.write("""
    This histogram visualizes the bounding box dimensions (width and height) in pixels, showing their frequency distribution. 

    ### Key Observations:
    - **Peak at Small Dimensions:** Most bounding boxes have small widths and heights, suggesting a prevalence of small objects in the dataset.
    - **Long Tail:** The distribution extends to larger dimensions, but their frequency is significantly lower, indicating fewer large objects.
    - **Comparison of Width and Height:** Both dimensions follow a similar pattern, but specific details might reveal if one is consistently larger than the other (aspect ratio differences).
    """)

# Object Density Heatmaps
elif section == "Object Density Heatmap Analysis":
    st.header("Object Density Heatmap Analysis")
    st.write("Object Density Heatmaps are visual representations that show the spatial distribution of objects within images. These heatmaps highlight regions with high or low object density, \
             providing insights into where objects are typically located. In object detection datasets, analyzing object density heatmaps helps identify biases, such as whether objects are concentrated in specific areas (e.g., roads or sidewalks) or uniformly distributed. This information is valuable for ensuring balanced model training and improving detection accuracy in real-world scenarios.")
    # Image dimensions (assuming all images are the same size)
    img_width, img_height = 1280, 720
    heatmap = np.zeros((img_height, img_width))

    for item in data:
        for label in item["labels"]:
            if "box2d" in label:
                bbox = label["box2d"]
                x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
                heatmap[y1:y2, x1:x2] += 1

    # Normalize heatmap for visualization
    heatmap = heatmap / heatmap.max()

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(heatmap, cmap="hot", interpolation="nearest")
    ax.set_title("Object Density Heatmap")
    ax.axis("off")
    
    st.pyplot(fig)
    st.subheader("Anomaly Insights:")

    st.write("""
            ### High-Density Region (Center):
            - The heatmap indicates a significantly high object density along the horizontal center, depicted by the bright yellow region.
            - This suggests that most objects in the dataset are concentrated in a central horizontal band, likely reflecting roadways or similar scenes.

            ### Low-Density Regions (Top and Bottom):
            - The darker red and black areas at the top and bottom indicate low object density.
            - This could imply fewer objects in regions like the sky (top) or ground (bottom), possibly introducing bias in object detection models for such regions.

            ### Potential Dataset Bias:
            - The central concentration could indicate a bias towards datasets with a primary focus on road-level scenes, limiting generalization to other environments or perspectives.

            ### Spatial Imbalance:
            - The heatmap lacks uniformity, with minimal object density in the peripheral areas.
            - This imbalance could result in models underperforming in edge-case scenarios where objects are located in these regions.
            """)


# Spatial Heatmap Analysis
elif section == "Advanced Spatial Heatmap Analysis":
    st.header("Advanced Spatial Heatmap Analysis")
    st.write("Advanced Spatial Heatmaps provide detailed visualizations of spatial patterns for individual object categories within a dataset. These heatmaps highlight where specific objects (e.g., cars, pedestrians, traffic lights) are most frequently located in the image space.\
              By analyzing these patterns, researchers can detect biases, such as certain objects being concentrated in specific regions (e.g., cars near the bottom or traffic lights at the top). This insight helps in refining object detection models to handle diverse spatial distributions effectively.")
    img_width, img_height = 1280, 720  # Assuming image resolution
    


    # Initialize heatmaps
    category_heatmaps = {category: np.zeros((img_height, img_width)) for category in categories}

    # Populate heatmaps
    for item in data:
        for label in item["labels"]:
            if label["category"] in categories and "box2d" in label:
                bbox = label["box2d"]
                x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
                category_heatmaps[label["category"]][y1:y2, x1:x2] += 1

    # Normalize and visualize heatmaps
    for category in categories:
        heatmap = category_heatmaps[category]
        if heatmap.max() != 0:
            heatmap /= heatmap.max()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(heatmap, cmap="hot", interpolation="nearest")
        ax.set_title(f"Spatial Heatmap for {category.capitalize()}")
        ax.axis("off")
        st.pyplot(fig)




elif section == "Category Correlation Analysis":
    st.header("Category Correlation Analysis")
    st.write("Correlation Analysis explores relationships between different attributes and categories in a dataset. For example, it examines how object categories (e.g., cars, pedestrians) are influenced by environmental factors such as weather, time of day, or scene type. By identifying correlations, this analysis helps uncover patterns, such as cars being more prevalent in highways or pedestrians being common in urban settings, enabling more informed dataset preparation and model training.")
    correlation_data = []

    for item in data:
        for label in item["labels"]:
            category = label["category"]
            weather = item["attributes"]["weather"]
            timeofday = item["attributes"]["timeofday"]
            scene = item["attributes"]["scene"]
            correlation_data.append({"category": category, "weather": weather, "timeofday": timeofday, "scene": scene})

    correlation_df = pd.DataFrame(correlation_data)

    # Plot category vs weather
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(x="weather", hue="category", data=correlation_df, ax=ax)
    ax.set_title("Category Distribution Across Weather Conditions")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    # for container in ax.containers:
    #     ax.bar_label(container, label_type='edge', rotation=90, padding=3)
    st.pyplot(fig)

    # Plot category vs time of day
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(x="timeofday", hue="category", data=correlation_df, ax=ax)
    ax.set_title("Category Distribution Across Time of Day")
    st.pyplot(fig)

    # Plot category vs Scene
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(x="scene", hue="category", data=correlation_df, ax=ax)
    ax.set_title("Category Distribution Across Scene")
    st.pyplot(fig)


elif section == "Annotations Quality Analysis":

    st.header("Annotations Quality Analysis")
    st.write(f"Annotations quality analysis evaluates the reliability and consistency of labeled data in object detection datasets. By identifying anomalies such as extremely small bounding boxes, misaligned labels, or missing annotations, it ensures the dataset's integrity and usability. For example, detecting an unusually high number of small bounding boxes for categories like **traffic light** may highlight issues with scale, sensor resolution, or labeling errors. This step is crucial for improving model performance by refining training data.")

    # Collect bounding box stats
    bbox_data = []
    for item in data:
        for label in item["labels"]:
            if "box2d" in label:
                bbox = label["box2d"]
                width = bbox["x2"] - bbox["x1"]
                height = bbox["y2"] - bbox["y1"]
                scene = item["attributes"]["scene"]
                bbox_data.append({"category": label["category"], "width": width, "height": height,"scene":scene})

    # Convert to DataFrame
    bbox_df = pd.DataFrame(bbox_data)

    # Detect small bounding boxes
    bbox_df["area"] = bbox_df["width"] * bbox_df["height"]
    small_bboxes = bbox_df[bbox_df["area"] < 50]  # Threshold for small bounding boxes

    print(f"Number of small bounding boxes: {len(small_bboxes)}")
    print(small_bboxes.head())

    small_bboxes_count = small_bboxes["category"].value_counts()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=small_bboxes_count.index, y=small_bboxes_count.values, palette="coolwarm")
    ax.set_title("Count of Small Bounding Boxes by Category")
    ax.set_xlabel("Category")
    ax.set_ylabel("Number of Small Bounding Boxes")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x="width", y="height", hue="category", data=bbox_df, alpha=0.7)
    ax.axvline(50, color="red", linestyle="--", label="Small Threshold")
    ax.axvline(50, color="red", linestyle="--")
    ax.set_title("Scatter Plot of Bounding Box Dimensions")
    ax.set_xlabel("Width")
    ax.set_ylabel("Height")
    ax.legend()
    st.pyplot(fig)

    # Object Overlapping chart
    def compute_iou(box1, box2):
        """
        Compute Intersection over Union (IoU) between two bounding boxes.
        Each box is a dictionary with x1, y1, x2, y2 keys.
        """
        x1 = max(box1['x1'], box2['x1'])
        y1 = max(box1['y1'], box2['y1'])
        x2 = min(box1['x2'], box2['x2'])
        y2 = min(box1['y2'], box2['y2'])

        # Intersection area
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)

        # Areas of both boxes
        box1_area = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
        box2_area = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])

        # Union area
        union_area = box1_area + box2_area - inter_area

        # IoU
        return inter_area / union_area if union_area > 0 else 0

    def compute_overlap_matrix(data, iou_threshold=0.5):
        """
        Compute overlap matrix for categories based on IoU.
        """
        categories = set(label["category"] for item in data for label in item["labels"])
        category_list = sorted(categories)
        overlap_matrix = pd.DataFrame(0, index=category_list, columns=category_list)

        for item in data:
            labels = item["labels"]
            for i, label1 in enumerate(labels):
                for j, label2 in enumerate(labels):
                    if i != j:  # Don't compare a box with itself
                        iou = compute_iou(label1["box2d"], label2["box2d"])
                        if iou >= iou_threshold:
                            overlap_matrix[label1["category"]][label2["category"]] += 1

        # Normalize overlap counts to percentages
        for category in overlap_matrix.index:
            total_annotations = overlap_matrix.loc[category].sum()
            if total_annotations > 0:
                overlap_matrix.loc[category] = (overlap_matrix.loc[category] / total_annotations) * 100

        return overlap_matrix

    # Compute overlap matrix
    overlap_matrix = compute_overlap_matrix(data, iou_threshold=0.5)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(overlap_matrix, annot=True, fmt=".1f", cmap="YlGnBu", cbar=True)
    ax.set_title("Object Annotation Overlap (IoU >= 0.5)")
    ax.set_xlabel("Category")
    ax.set_ylabel("Category")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    # ax.tight_layout()
    st.pyplot(fig)

elif section=="Object Co-Occurrence Analysis":
    st.header("Object Co-Occurrence Analysis")
    st.write(f"Object co-occurrence analysis examines the relationships between different object categories within a dataset, identifying which objects frequently appear together in the same scene or context. This insight is valuable for understanding real-world associations, improving object detection models, and optimizing scene understanding tasks. For example, in traffic datasets, vehicles often co-occur with pedestrians or traffic lights, providing context-aware cues for autonomous systems.")
    category_counts = Counter()
    for item in data:
        for label in item["labels"]:
            category_counts[label["category"]] += 1

    # Create co-occurrence matrix
    categories = list(category_counts.keys())
    co_occurrence = {category: Counter() for category in categories}

    for item in data:
        image_categories = []
        for label in item["labels"]:
            if 'box2d' not in label:
                continue
            image_categories.append(label["category"])
        for cat1, cat2 in combinations(image_categories, 2):
            co_occurrence[cat1][cat2] += 1
            co_occurrence[cat2][cat1] += 1

    # Convert to a DataFrame
    co_occurrence_df = pd.DataFrame(co_occurrence).fillna(0)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(co_occurrence_df, annot=True, fmt=".0f", cmap="Blues", xticklabels=True, yticklabels=True)
    ax.set_title("Object Co-Occurrence Heatmap")
    st.pyplot(fig)
    
# Category-Based Viewer with Navigation
elif section == "Category-Based Viewer":
    st.header("Category-Based Viewer")

    filter_category = st.selectbox("Select a category to filter images", options=categories)

    filtered_items = [item for item in data if any(label["category"] == filter_category for label in item["labels"])]
    
    # Initialize the filtered index for the selected category in session state
    if filter_category not in st.session_state:
        st.session_state[filter_category] = 0

    def load_filtered_image(index):
        item = filtered_items[index]
        img_name = item["name"]
        img_path = os.path.join(image_path, img_name)
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            st.error(f"Image not found: {img_path}")
            return None

        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()

        filtered_labels = [label for label in item["labels"] if label["category"] == filter_category]
        for label in filtered_labels:
            box = label["box2d"]
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1 - 10), label["category"], fill="white", font=font)

        return img, item["attributes"]

    # Navigation buttons
    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        if st.button("Previous", key=f"{filter_category}_prev"):
            st.session_state[filter_category] = max(0, st.session_state[filter_category] - 1)
    with col3:
        if st.button("Next", key=f"{filter_category}_next"):
            st.session_state[filter_category] = min(len(filtered_items) - 1, st.session_state[filter_category] + 1)

    # Display filtered image and attributes
    if filtered_items:
        img, attributes = load_filtered_image(st.session_state[filter_category])
        if img:
            st.image(img, caption=f"Filtered Image {st.session_state[filter_category] + 1} of {len(filtered_items)} ({filter_category})", use_container_width=True)
            st.write(f"**Weather:** {attributes['weather']} | **Scene:** {attributes['scene']} | **Time of Day:** {attributes['timeofday']}")
    else:
        st.write("No images found for the selected category.")
