from datasets import load_dataset, concatenate_datasets
import json

# Load the MRPC dataset
dataset = load_dataset("glue", "mrpc")

# Save the dataset locally
dataset.save_to_disk("mrpc_dataset")

# Function to convert to loose JSON
def convert_to_loose_json(dataset, output_file):
    with open(output_file, 'w') as f:
        for i, example in enumerate(dataset):
            json_line = {
                "src": "MRPC",
                "text": example['sentence1'] + ' [SEP] ' + example['sentence2'],
                "type": "Eng",
                "id": str(i),
                "title": f"MRPC Example {i}"
            }
            f.write(json.dumps(json_line) + '\n')

# Combine train, validation, and test sets
# combined_dataset = concatenate_datasets([dataset["train"], dataset["validation"], dataset["test"]])

# Convert combined dataset to loose JSON
convert_to_loose_json(dataset["train"], 'mrpc_train.json')

print(f"Combined dataset size: {len(combined_dataset)}")
