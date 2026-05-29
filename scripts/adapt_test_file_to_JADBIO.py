import pandas as pd
from pathlib import Path

train_path = Path(r"C:\Users\hp\Desktop\stage etis\jadbio_all_feature_types.csv")
test_path  = Path(r"C:\Users\hp\Desktop\stage etis\open_dataset\jadbio_csvs\jadbio_animal_musk_test_real.csv")

train = pd.read_csv(train_path, sep=";")
test  = pd.read_csv(test_path,  sep=";")

# Bach nforcer koul column ftest file tmacthe type of the corresponding column in train file
for col in test.columns:
    if col in train.columns:
        try:
            test[col] = test[col].astype(train[col].dtype)
        except Exception as e:
            print(f"Could not convert '{col}': {e}")

out = test_path.parent / (test_path.stem + "_fixed4.csv")
test.to_csv(out, sep=";", index=False)
print(f"Saved: {out.name}")