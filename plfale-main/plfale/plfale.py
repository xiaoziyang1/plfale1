import numpy as np
from .BasePLECOC import BasePLECOC
from sklearn.metrics import f1_score,roc_auc_score,accuracy_score
import plfale.encoder as ec
from plfale.feature_select import lookahead_feature_select
import copy

class plfale(BasePLECOC):
    def __init__(self, params):
        BasePLECOC.__init__(self, params)
        self.max_iter = self.params.get("max_iter", 20000)
        self.num_class = None
        self.codingLength = None
        self.min_num_tr = None
        self.coding_matrix = None
        self.models = None
        self.performance_matrix = None
        self.islibsvm = None
        self.classify_scores = None
        self.distance_value = None
        self.common_value = None
        self.error_value = None
        self.bin_pre = None
        self.feature_index = []
        self.auc = []
        self.acc = []
        self.base_classifier = self.params.get("classifier")
        self.confusion_matrix = None

    def create_coding_matrix(self, tr_data, tr_labels):
        num_tr = tr_data.shape[0]
        self.num_class = tr_labels.shape[0]
        self.codingLength = int(np.ceil(10 * np.log2(self.num_class)))
        self.min_num_tr = int(np.ceil(0.1 * num_tr))

        coding_matrix = []
        counter = 0
        tr_pos_idx = []
        tr_neg_idx = []
        # sample selection
        next_layer = np.full(tr_data.shape[0], True)
        for i in range(self.max_iter):
            if tr_data[next_layer].shape[0] < tr_labels.shape[0]:
                next_layer = ~next_layer
            next_layer = self.fill_zero_label(tr_labels, next_layer)
            tr_data_i = tr_data[next_layer]
            tr_labels_i = tr_labels[:, next_layer]
            if tr_data[next_layer].shape[0] < tr_labels.shape[0]:
                next_layer = ~next_layer
            # encode column
            label_list_i ,combine_label_set= ec.encoder(tr_data_i, tr_labels_i)
            positive_label = label_list_i[0]
            negative_label = label_list_i[1]

            tmpcode = np.zeros(self.num_class)

            for raw_i in range(self.num_class):
                if positive_label[raw_i] == 1 and negative_label[raw_i] == 0:
                    tmpcode[raw_i] = 1
                elif negative_label[raw_i] == 1 and positive_label[raw_i] == 0:
                    tmpcode[raw_i] = -1
                elif positive_label[raw_i] == 0 and negative_label[raw_i] == 0:
                    tmpcode[raw_i] = 0
            next_layer = ~next_layer
            tmpcode = np.int8(tmpcode)

            if self.check_column(coding_matrix, tmpcode):
                if counter > self.codingLength/3:
                    tmpcode = self.reverse_part_code(coding_matrix, tmpcode)
            else:
                continue
            tmp_pos_idx = []
            tmp_neg_idx = []
            # train data extraction
            for j in range(num_tr):
                if np.all((np.multiply(tr_labels[:, j], tmpcode)) == tr_labels[:, j]):
                    tmp_pos_idx.append(j)
                elif np.all((np.multiply(tr_labels[:, j], -tmpcode)) == tr_labels[:, j]):
                    tmp_neg_idx.append(j)
                else:
                    next_layer[j] = True

            num_pos = len(tmp_pos_idx)
            num_neg = len(tmp_neg_idx)
            if (num_pos+num_neg >= self.min_num_tr) and (num_pos >= 5) and (num_neg >= 5) \
                    and self.check_column(coding_matrix, tmpcode):
                counter = counter + 1
                tr_pos_idx.append(tmp_pos_idx)
                tr_neg_idx.append(tmp_neg_idx)
                coding_matrix.append(tmpcode)

            if counter >= self.codingLength and self.check_row(coding_matrix):
                self.codingLength = counter
                break
        if counter < self.codingLength:
            # raise ValueError('The required codeword length %s not satisfied', str(self.codingLength))
            self.codingLength = counter
            if counter == 0:
                raise ValueError('Empty coding matrix')
        coding_matrix = np.array(coding_matrix).transpose()
        return coding_matrix, tr_pos_idx, tr_neg_idx

    def create_base_models(self, tr_data, tr_pos_idx, tr_neg_idx):
        train_times = np.zeros(tr_data.shape[0])
        models = []
        for i in range(self.codingLength):
            train_times[tr_pos_idx[i]] = train_times[tr_pos_idx[i]] + 1
            train_times[tr_neg_idx[i]] = train_times[tr_neg_idx[i]] + 1
            pos_inst = tr_data[tr_pos_idx[i]]
            neg_inst = tr_data[tr_neg_idx[i]]
            tr_inst = np.vstack((pos_inst, neg_inst))
            tr_labels = np.hstack((np.ones(len(pos_inst)), -np.ones(len(neg_inst))))
            feature_index = lookahead_feature_select(tr_inst, tr_labels, self.base_classifier(self.params))
            classifier = self.base_classifier(self.params)
            classifier.fit(tr_inst[:, feature_index].tolist(), tr_labels.tolist())
            p_labels, p_vals = classifier.predict_proba(tr_inst[:, feature_index].tolist())
            p_labels = np.array(p_labels)
            self.auc.append(roc_auc_score(tr_labels, p_labels))
            self.acc.append(accuracy_score(tr_labels, p_labels))
            models.append(classifier)
            self.feature_index.append(feature_index)
        return models

    def create_performance_matrix(self, tr_data, tr_labels):
        performance_matrix = np.zeros((self.num_class, self.codingLength))
        scores = []
        for i in range(self.codingLength):
            model = self.models[i]
            p_labels = model.predict(tr_data[:, self.feature_index[i]].tolist())
            p_labels = [int(i) for i in p_labels]
            for j in range(self.num_class):
                label_class_j = np.array(p_labels)[tr_labels[j, :] == 1]
                performance_matrix[j, i] = sum(np.abs(label_class_j[label_class_j ==
                                         self.coding_matrix[j, i]]))/label_class_j.shape[0]
            t = performance_matrix.sum(axis=1)
        return performance_matrix / np.transpose(np.tile(t, (performance_matrix.shape[1], 1))), scores

    def fit(self, tr_data, tr_labels):
        if tr_data.shape[0] < 15000:
            self.islibsvm = True
        else:
            self.islibsvm = False
        self.coding_matrix, tr_pos_idx, tr_neg_idx = self.create_coding_matrix(tr_data, tr_labels)
        self.models = self.create_base_models(tr_data, tr_pos_idx, tr_neg_idx)
        self.performance_matrix, self.classify_scores = self.create_performance_matrix(tr_data, tr_labels)
    #loss weight experience decoding
    def predict(self, ts_data):
        bin_pre = None
        decision_pre = None
        for i in range(self.codingLength):
            model = self.models[i]
            p_labels, p_vals = model.predict_proba(ts_data[:, self.feature_index[i]].tolist())
            bin_pre = p_labels if bin_pre is None else np.vstack((bin_pre, p_labels))
            decision_pre = np.array(p_vals).T if decision_pre is None else np.vstack((decision_pre, np.array(p_vals).T))
        output_value = np.zeros((self.num_class, ts_data.shape[0]))
        common_value = np.zeros((self.num_class, ts_data.shape[0]))
        error_value = np.zeros((self.num_class, ts_data.shape[0]))
        for i in range(ts_data.shape[0]):
            bin_pre_tmp = bin_pre[:, i]
            decision_pre_tmp = decision_pre[:, i]
            for j in range(self.num_class):
                code = self.coding_matrix[j, :]
                common = np.int8(bin_pre_tmp == code) * self.performance_matrix[j, :] / np.exp(np.abs(decision_pre_tmp))
                error = np.int8(bin_pre_tmp != code) * self.performance_matrix[j, :] * np.exp(np.abs(decision_pre_tmp))
                output_value[j, i] = -sum(common)-sum(error)
        self.distance_value = -1 * output_value
        self.common_value = common_value
        self.error_value = error_value
        pre_label_matrix = np.zeros((self.num_class, ts_data.shape[0]))
        for i in range(ts_data.shape[0]):
            idx = output_value[:, i] == max(output_value[:, i])
            pre_label_matrix[idx, i] = 1
        return pre_label_matrix

    def score(self, X, y_test):
        y_predict = self.predict(X)
        result = y_predict * y_test
        labels = list(range(y_test.shape[0]))
        p_label = np.dot(labels, y_predict)
        true_label = np.dot(labels, y_test)
        fscore = f1_score(true_label, p_label, labels, average='macro')
        return y_predict, sum(np.sum(result, axis=0))/len(X), fscore

    def check_row(self, matrix):
        if matrix:
            matrix = np.array(matrix).transpose()
            matrix_unique = np.unique(matrix, axis=0)
            size = matrix_unique.shape[0]
            if size < self.num_class:
                return False
            else:
                return True
        else:
            return False

    def check_column(self, matrix, column):
        code = column
        matrix = np.array(matrix).transpose()
        if not matrix.size:
            return True
        for i in range(matrix.shape[1]):
            if (matrix[:, i].reshape(1, -1) == code).all() or ((matrix[:, i].reshape(1, -1)) == -code).all():
                # print("same column")
                return False
        return True


    def fill_zero_label(self, tr_labels, next_layer):
        ratio = np.sum(next_layer)/tr_labels.shape[1]
        train_label = tr_labels[:, next_layer]
        train_label_sum = np.sum(train_label, axis=1)
        zero_label = []
        for i in range(len(train_label_sum)):
            if train_label_sum[i] == 0:
                zero_label.append(i)
        p_label = np.array(tr_labels).transpose()
        for label_i in zero_label:
            ratio_zero_mask = self.get_zero_data_by_ratio(label_i, p_label, ratio)
            next_layer = next_layer | ratio_zero_mask
            # next_layer = next_layer | p_label[:, label_i] == 1
        return next_layer

    def get_zero_data_by_ratio(self, label_i, p_label, ratio):
        index_ = np.array(range(p_label.shape[0]))
        index_mask = index_[p_label[:, label_i] == 1]
        np.random.shuffle(index_mask)
        size = int(p_label.shape[0] * ratio)
        ratio_zero_index = index_mask[:size]
        tmp_mask = np.full(p_label.shape[0], False)
        tmp_mask[ratio_zero_index] = True
        ratio_zero_mask = tmp_mask
        return ratio_zero_mask

    def get_confusion_matrix(self, y_test, y_pred):
        self.confusion_matrix = np.zeros((self.coding_matrix. shape[0],self.coding_matrix.shape[0]))
        y_test = np.array(y_test).astype(int)
        y_pred = np.array(y_pred).astype(int)
        for i, j in zip(y_test, y_pred):
            self.confusion_matrix[i][j] += 1
        return self.confusion_matrix

    def reverse_part_code(self, matrix, column):
        reverse_column = copy.deepcopy(column)
        min_hamming_distance = float("inf")
        proto_matrix = copy.deepcopy(matrix)
        proto_matrix.append(column)
        proto_matrix = np.array(proto_matrix).transpose()
        min_distance_class = []
        for i in range(self.num_class):
            for j in range(self.num_class):
                if j > i:
                    tmp_distance = self.hamming_distance(proto_matrix[i, :], proto_matrix[j, :])
                    if tmp_distance < min_hamming_distance:
                        min_hamming_distance = tmp_distance
        for k in range(self.num_class):
            tmp_distance = self.cal_min_hamming_distance(proto_matrix, k)
            if tmp_distance == min_hamming_distance:
                min_distance_class.append(k)
        min_distance_class = np.array(min_distance_class)
        # 将最小的行进行乱序排序，避免处于前面的class偏向-1
        np.random.shuffle(min_distance_class)
        # min_distance_class = np.unique(np.array(min_distance_class))
        for min_i in min_distance_class:
            tmp_distance = self.cal_min_hamming_distance(proto_matrix, min_i)
            if tmp_distance <= min_hamming_distance:
                proto_matrix[min_i][proto_matrix.shape[1]-1] = -proto_matrix[min_i][proto_matrix.shape[1]-1]
                reverse_column[min_i] = -reverse_column[min_i]
        return reverse_column

    def hamming_distance(self, x, y):
        x = np.array(x).reshape(1, -1)
        y = np.array(y).reshape(1, -1)
        assert len(x) == len(y)
        distance = np.sum(((1 - np.sign(x * y))/2))
        return distance

    def cal_min_hamming_distance(self, proto_matrix, i):
        min_hamming_distance = float("inf")
        for j in range(self.num_class):
            if i != j:
                tmp_distance = self.hamming_distance(proto_matrix[i, :], proto_matrix[j, :])
                if tmp_distance < min_hamming_distance:
                    min_hamming_distance = tmp_distance
        return min_hamming_distance

