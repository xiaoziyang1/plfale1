import os
import copy
import pandas as pd
import numpy as np
import random as ran
from scipy import io
from sklearn import preprocessing




def norm_data(data):
    data = preprocessing.scale(data.astype('float64'))
    return data


def load_data(target_file):
    if os.path.exists(target_file):
        mat = io.loadmat(target_file)
        data = mat['data']
        data = norm_data(data)
        targets = mat['target']
        if type(targets) != np.ndarray:
            targets = targets.toarray()
        targets = targets.astype(np.int)
        partial_targets = mat['partial_target']
        if type(partial_targets) != np.ndarray:
            partial_targets = partial_targets.toarray()
        partial_targets = partial_targets.astype(np.int)

    else:
        raise FileNotFoundError(target_file)
    return data, targets, partial_targets


