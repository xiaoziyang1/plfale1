# from ecoc import *
from plfale.plfale import plfale
from sklearn.model_selection import StratifiedKFold
from learner import *
from tools import Uci_to_mat
import numpy as np
import os
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


k_fold = 10
base_learners = [SVM]
base_path = "mat"
if __name__ == '__main__':
    real_datasets = [
        "lost.mat"
        # "MSRCv2.mat",
        # "BirdSong.mat",
        # "Mirflickr.mat",
        # "Yahoo! News.mat",
        # "Soccer Player.mat",
        # "FG-NET.mat"
    ]
    all_task = []
    skf = StratifiedKFold(n_splits=k_fold)
    # 分别加载数据
    for real_dataset in real_datasets:
        print(real_dataset)
        result = []
        path = os.path.join(base_path, real_dataset)
        data, true_labels, partial_labels = Uci_to_mat.load_data(path)
    # 每个数据集重复运行repetitions次
        labels = list(range(0, true_labels.shape[0]))
        y = np.dot(labels, true_labels)
        i = 0
        # 每次运行都做k_fold折交叉验证
        for tr_index, ts_index in skf.split(data, y):
            i = i + 1
            tr_data, ts_data = data[tr_index, :], data[ts_index, :]
            tr_label, ts_label = partial_labels[:, tr_index], true_labels[:, ts_index]
            ecoc_models = plfale(params={"classifier": SVM, "svm_param": '-t 2 -c 1 -q', "gamma" :3})
            ecoc_models.fit(tr_data, tr_label)
            pr_label = np.dot(list(range(0, tr_label.shape[0])), ecoc_models.predict(ts_data))
            tr_ts_label = np.dot(list(range(0, tr_label.shape[0])), ts_label)
            result_tmp = accuracy_score(y_pred=pr_label, y_true=tr_ts_label)
            print(result_tmp)
    print('finish')


