import numpy as np
from learner import *
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE,VarianceThreshold,chi2,f_classif,mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
import heapq
import copy
import os
# from tools import Uci_to_mat
from sklearn.preprocessing import MinMaxScaler

# greedy feature selection
def lookahead_feature_select(data, label, classifier=SVM(params={"svm_param": '-t 2 -c 1 -q'})):
    k_fold = 5
    skf = StratifiedKFold(n_splits=k_fold)
    feature_num = []
    feature_rank = None
    for tr_index, ts_index in skf.split(data, label):
        # print("feature_rank")
        tr_data, ts_data = data[tr_index, :], data[ts_index, :]
        tr_label, ts_label = label[tr_index], label[ts_index]

        auc = 0
        features_to_keep = 0
        ranking = rfe_ranking(tr_data, tr_label)
        for k in range(tr_data.shape[1]):
            tmp_feature_index = heapq.nsmallest(k+1, range(len(ranking)), ranking.take)
            classifier.fit(tr_data[:, tmp_feature_index].tolist(), tr_label.tolist())
            pr_label = classifier.predict(ts_data[:, tmp_feature_index].tolist())
            tmp_auc = roc_auc_score(y_true=ts_label, y_score=pr_label)
            if tmp_auc >= auc:
                features_to_keep = k+1
                auc = tmp_auc

        feature_num.append(features_to_keep)
        if feature_rank is None:
            feature_rank = copy.deepcopy(ranking)
        else:
            feature_rank = np.vstack((feature_rank, ranking))
    feature_num = np.array(feature_num)
    final_feature_num = int(np.rint(np.mean(feature_num)))
    feature_rank = np.sum(feature_rank, axis=0)
    final_feature_index = heapq.nsmallest(final_feature_num, range(len(feature_rank)), feature_rank.take)
    return final_feature_index


def rfe_ranking(data, label):
    selector = RFE(estimator=SVC(kernel='linear', C=1, gamma='auto'), n_features_to_select=1, step=1)
    selector = selector.fit(data, label)
    ranking = selector.ranking_
    return ranking


def vt_ranking(data):
    sel = VarianceThreshold()
    sel.fit(data)
    sort_index = np.argsort(sel.variances_)
    ranking = np.zeros(len(sort_index))
    rank = 1
    sort_index = sort_index[::-1]
    for feature_i in sort_index:
        ranking[feature_i] = rank
        rank += 1
    return ranking


# 输入的数据有负数，用不了卡方
def chi2_ranking(data, label):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    feature_score, p_val = chi2(data, label)
    tmp_index = np.argsort(feature_score)
    ranking = np.zeros(len(tmp_index))
    rank = 1
    sort_index = tmp_index[::-1]
    for feature_i in sort_index:
        ranking[feature_i] = rank
        rank += 1
    return ranking


def f_classif_ranking(data, label):
    feature_score, p_val = f_classif(data, label)
    tmp_index = np.argsort(feature_score)
    ranking = np.zeros(len(tmp_index))
    rank = 1
    sort_index = tmp_index[::-1]
    for feature_i in sort_index:
        ranking[feature_i] = rank
        rank += 1
    return ranking


def mutual_info_classif_ranking(data, label):
    feature_score = mutual_info_classif(data, label, discrete_features='auto', n_neighbors=3, copy=True, random_state=None)
    tmp_index = np.argsort(feature_score)
    ranking = np.zeros(len(tmp_index))
    rank = 1
    sort_index = tmp_index[::-1]
    for feature_i in sort_index:
        ranking[feature_i] = rank
        rank += 1
    return ranking


