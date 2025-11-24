import h5py
import numpy as np

def create_HDF5_file(vector_length,name="hdf_file",data_name="vectors",label_name="coordinates",label_size=2,label_type="float32"):
    with h5py.File(f"{name}.h5", 'w') as f:
        # Create a resizable dataset, initially empty (0 rows, vector_length columns)
        dset = f.create_dataset(
            data_name,
            shape=(0, vector_length),
            maxshape=(None, vector_length),
            chunks=True,
            dtype='float32'
        )
        labels = f.create_dataset( #
            label_name,
            shape=(0,label_size),
            maxshape=(None,label_size),
            chunks=True,
            dtype=label_type
        )
        dset = f.create_dataset(
            "dict_idx",
            shape=(0,1),
            maxshape=(None, 1),
            chunks=True,
            dtype='int'
        )
def open_HDF5(filename):
    '''MUST close with file.close()'''
    file = h5py.File(filename, 'r+')
    return file

def append_HDF5(vectors_to_add, label_to_add, file, data_name="vectors", label_name="coordinates", dict_idx=None):
    # Resize and append vectors
    vectors = file[data_name]
    vectors.resize(vectors.shape[0] + vectors_to_add.shape[0], axis=0)
    vectors[-vectors_to_add.shape[0]:] = vectors_to_add

    # Resize and append coordinates/labels
    coordinates = file[label_name]
    coordinates.resize(coordinates.shape[0] + label_to_add.shape[0], axis=0)
    coordinates[-label_to_add.shape[0]:] = label_to_add

    if dict_idx is not None:
        if "dict_idx" not in file:
            maxshape = (None,)
            file.create_dataset("dict_idx", data=dict_idx, maxshape=maxshape, dtype="int64")
        else:
            idx_ds = file["dict_idx"]
            idx_ds.resize(idx_ds.shape[0] + len(dict_idx), axis=0)
            idx_ds[-len(dict_idx):] = dict_idx



def get_size(path):
    with h5py.File(path, 'r') as f:
        f.visititems(print_shapes)

def print_shapes(name, obj):#used fpr "get_size"
        if isinstance(obj, h5py.Dataset):
            print(f"{name}: shape={obj.shape}, dtype={obj.dtype}")