import numpy as np

data = np.load('data/cse413_protein_group_split_seed413.npz', allow_pickle=True)

print("\n--- NPZ Data Info ---")
for key in data.files:
    val = data[key]
    if hasattr(val, 'shape'):
        print(f"{key}: {val.shape}")
    else:
        print(f"{key}: {type(val)}")
print("---------------------\n")