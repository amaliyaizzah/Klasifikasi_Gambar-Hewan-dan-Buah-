import h5py

def print_h5_structure(g, indent=0):
    for key in g.keys():
        item = g[key]
        print("  " * indent + f"{key} ({type(item)})")
        if isinstance(item, h5py.Group):
            print_h5_structure(item, indent + 1)
        elif isinstance(item, h5py.Dataset):
            print("  " * (indent + 1) + f"Shape: {item.shape}, Dtype: {item.dtype}")

with h5py.File('model_hewan_buah.h5', 'r') as file:
    print("📂 Struktur file HDF5:")
    print_h5_structure(file)
