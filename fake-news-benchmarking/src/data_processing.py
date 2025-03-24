import pandas as pd
import os

# Define dataset folder path
dataset_folder = "C:/Users/KIIT/Downloads/MINORR/fake_news/fake-news-benchmarking/datasets"

# Check if dataset folder exists
if not os.path.exists(dataset_folder):
    print(f"❌ Error: Dataset folder '{dataset_folder}' not found!")
    exit()

# Function to load dataset files
def load_data(file_path):
    try:
        if file_path.endswith(".csv"):
            return pd.read_csv(file_path)
        elif file_path.endswith(".tsv"):
            return pd.read_csv(file_path, delimiter="\t")
        elif file_path.endswith(".json"):
            return pd.read_json(file_path)
        else:
            print(f"⚠️ Skipping unsupported file: {file_path}")
            return None
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return None

# Function to preprocess dataset files
def preprocess_data(dataset_folder):
    all_files = []
    
    # Loop through dataset folder
    for root, _, files in os.walk(dataset_folder):
        for file in files:
            file_path = os.path.join(root, file)
            print(f"📂 Found file: {file_path}")  # Debug print

            # Load data
            df = load_data(file_path)
            if df is not None:
                all_files.append(df)

    # Check if any files were loaded
    if not all_files:
        print("❌ No valid dataset files found! Check dataset folder path and file formats.")
        exit()

    # Print loaded file details
    print(f"✅ Loaded {len(all_files)} files:")
    for i, file in enumerate(all_files):
        print(f"📄 File {i+1}: {file.shape}, Columns: {list(file.columns)}")

    # Merge all data
    full_df = pd.concat(all_files, ignore_index=True)
    return full_df

# Run preprocessing
df = preprocess_data(dataset_folder)
print("✅ Preprocessing complete. Final dataset shape:", df.shape)
