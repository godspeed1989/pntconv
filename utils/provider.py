"""
    data provider
"""
import os
import h5py
import numpy as np

def getDataFiles(list_filename):
    """ Get data list file content
        Input:
          list_filename: list file path
        Output:
          data file path list
    """
    return [line.rstrip() for line in open(list_filename)]

def loadDataFile(h5_filename, dataset_path):
    """ Load data and label from hdf5 file.
        Input:
          h5_filename: file name
          dataset_path: file path
        Output:
          data, label
    """
    f = h5py.File(os.path.join(dataset_path, h5_filename), 'r')
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx