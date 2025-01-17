### **Sidebar Navigation Sections**

#### 1\. **Dataset Overview**

-   **Description:** Provides a high-level summary of the dataset.
-   **Details:**
    -   Displays the total number of images in the dataset.
    -   Shows the structure of the dataset by viewing the first JSON record.
-   **Purpose:** To help users quickly understand the dataset's size and format.

* * * * *

#### 2\. **Object Category Distribution**

-   **Description:** Visualizes the count of objects across categories.
-   **Details:**
    -   Bar chart showing the frequency of each object category.
    -   Counts are displayed above the bars for better clarity.
-   **Purpose:** To identify imbalances in the dataset and understand object diversity.

* * * * *

#### 3\. **Weather Conditions**

-   **Description:** Analyzes the distribution of images across different weather conditions.
-   **Details:**
    -   Pie chart showing the proportions of weather conditions (e.g., Clear, Foggy, Rainy).
    -   Anomaly insights highlight underrepresented or overrepresented conditions.
-   **Purpose:** To understand environmental biases and their potential impact on model performance.

* * * * *

#### 4\. **Time of Day**

-   **Description:** Analyzes the distribution of images across different times of day.
-   **Details:**
    -   Bar chart showing counts for Daytime, Night, Dawn/Dusk, and Undefined.
    -   Anomaly insights to detect imbalances, such as dominance of daytime images.
-   **Purpose:** To highlight lighting condition diversity and its effect on model training.

* * * * *

#### 5\. **Scene Distribution**

-   **Description:** Examines the distribution of images across scene types.
-   **Details:**
    -   Bar chart showing counts for scenes like City, Residential, and Highway.
    -   Highlights the variety of scenes in the dataset.
-   **Purpose:** To evaluate scene diversity and identify overrepresented environments.

* * * * *

#### 6\. **Bounding Box Statistics**

-   **Description:** Provides statistical analysis of bounding box dimensions.
-   **Details:**
    -   Metrics such as average, minimum, and maximum box sizes.
    -   Histogram visualizing the distribution of bounding box areas.
-   **Purpose:** To analyze object size variability and its implications for detection models.

* * * * *

#### 7\. **Object Density Heatmap Analysis**

-   **Description:** Visualizes the spatial distribution of objects across images.
-   **Details:**
    -   Heatmaps showing where objects frequently appear in images.
    -   Analysis for specific categories like cars or pedestrians.
-   **Purpose:** To detect spatial biases and identify common object locations.

* * * * *

#### 8\. **Advanced Spatial Heatmap Analysis**

-   **Description:** Provides detailed spatial heatmaps for multiple categories.
-   **Details:**
    -   Normalized heatmaps for objects like traffic signs, bikes, and traffic lights.
    -   Visual overlays to compare spatial distributions across categories.
-   **Purpose:** To conduct a deeper analysis of spatial patterns and category relationships.

* * * * *

#### 9\. **Category Correlation Analysis**

-   **Description:** Analyzes relationships between object categories and environmental attributes.
-   **Details:**
    -   Charts showing how categories correlate with weather conditions and times of day.
    -   Highlights potential biases or trends in object-environment interactions.
-   **Purpose:** To uncover hidden patterns and assess data balance.

* * * * *

#### 10\. **Annotations Quality Analysis**

-   **Description:** Evaluates the quality of annotations in the dataset.
-   **Details:**
    -   Checks for missing or incomplete annotations.
    -   Flags overlapping bounding boxes and incorrect label assignments.
-   **Purpose:** To ensure dataset consistency and identify areas needing correction.

* * * * *

#### 11\. **Object Co-Occurrence Analysis**

-   **Description:** Studies the co-occurrence of object categories within images.
-   **Details:**
    -   Heatmaps showing how often categories appear together.
    -   Insights into common object pairings (e.g., cars and traffic lights).
-   **Purpose:** To identify category relationships and inform model design.

* * * * *

#### 12\. **Category-Based Viewer**

-   **Description:** Allows users to filter and explore images containing a specific category.
-   **Details:**
    -   Select a category (e.g., Car, Person) to display relevant images.
    -   Navigate through filtered images using Previous/Next buttons.
    -   Displays bounding boxes for the selected category and image attributes (Weather, Scene, Time of Day).
-   **Purpose:** To enable targeted exploration of the dataset.