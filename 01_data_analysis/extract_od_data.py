# Extracting Object detection bounding box from BDD Dataset

import json
import os
import argparse

def filter_labels(input_path, output_path):
    try:
        # Check if the input file exists
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Load the JSON data
        with open(input_path, 'r') as f:
            data = json.load(f)

        # Filter labels with box2d
        for annotation in data:
            annotation['labels'] = [label for label in annotation.get('labels', []) if 'box2d' in label]

        # Save the cleaned data
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)

        print(f"Filtered data saved to {output_path}")

    except FileNotFoundError as e:
        print(e)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter BDD100K labels to include only those with box2d annotations.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the filtered JSON file.")

    args = parser.parse_args()

    # Call the filter function with provided arguments
    filter_labels(args.input, args.output)