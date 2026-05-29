import json
import pandas as pd

# 1. Load the JSON file
with open(r"C:\Users\hp\Desktop\stage etis\jadbio_data.json", 'r') as file:
    data = json.load(file)

# 2. Grab features from BOTH lists in the JSON
feature_info_list = data.get("featureInfo", [])
deduced_list = data.get("deduced", [])

# 3. Combine them into one single list
all_features_data = feature_info_list + deduced_list

# 4. Parse the combined list
parsed_features = []
for feature in all_features_data:
    parsed_features.append({
        "Feature_Name": feature.get("name"),
        "Assigned_Type": feature.get("type"),
        "Original_Index": feature.get("index") # Grabbing the index to keep things in order
    })

# 5. Export to CSV
df = pd.DataFrame(parsed_features)

# Sort by the original index so it matches your raw dataset's column order, then drop the index column to keep it clean
df = df.sort_values(by="Original_Index").drop(columns=["Original_Index"])

df.to_csv("jadbio_all_feature_types.csv", index=False)
print(f"Successfully exported {len(df)} features to jadbio_all_feature_types.csv")