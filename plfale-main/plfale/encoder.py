import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering



def encoder(data, tr_label):
    partial_label = np.array(tr_label).transpose()
    k = cal_appropriate_k(partial_label)
    cluster_labels = clustering(data, k)
    cluster_data, cluster_partial_label = group_cluster(data, partial_label, cluster_labels)
    cluster_centers = cal_cluster_center(cluster_data)
    label_distributions = cal_partial_label_set(cluster_partial_label)
    pure_label_set = cal_pure_label_set(label_distributions)
    final_label_set = pure_label_set

    separate_labels = combine_cluster_by_label_distance(label_distributions, cluster_centers)

    combine_label_set = cal_combine_list(separate_labels, final_label_set)

    final_label_set = distinguish_share_or_interfere_label(label_distributions, final_label_set,
                                                             separate_labels, combine_label_set)
    return final_label_set,combine_label_set


def cal_appropriate_k(partial_label):
    sum_partial_label = np.sum(partial_label, axis=0)
    k = partial_label.shape[1]
    for label_i_sum in sum_partial_label:
        if label_i_sum == 0:
            k = k-1
    return k

def group_cluster(data, partial_label, cluster_labels):
    mask = []
    cluster_data = []
    cluster_partial_label = []
    for label_i in np.unique(cluster_labels):
        mask.append(cluster_labels == label_i)
    for mask_i in mask:
        cluster_data.append(data[mask_i])
        cluster_partial_label.append(partial_label[mask_i])
    return cluster_data, cluster_partial_label


def cal_partial_label_set(cluster_partial_label):
    label_distributions = []
    for i in range(len(cluster_partial_label)):
        cluster_partial_label_i = cluster_partial_label[i]
        label_distribution_i = cal_label_distribution(cluster_partial_label_i)
        label_distributions.append(label_distribution_i)
    label_distributions = np.array(label_distributions)
    return label_distributions


def cal_label_distribution(partial_label):
    label_distribution = []
    for i in range(partial_label.shape[1]):
        mask = partial_label[:, i] == 1
        label_distribution.append(partial_label[mask].shape[0])
    label_distribution.append(partial_label.shape[0])
    label_distribution = np.array(label_distribution)
    return label_distribution


def cal_cluster_center(cluster_data):
    cluster_center = []
    for cluster_data_i in cluster_data:
        center_i = np.mean(cluster_data_i, axis=0)
        cluster_center.append(center_i)
    return cluster_center



def combine_cluster_by_label_distance(label_distributions, cluster_center):
    select_labels = list(range(label_distributions.shape[0]))
    # positive_labels 与negative_labels 里面都是
    positive_labels = list()
    negative_labels = list()
    positive_labels.append(find_max_data_cluster_distributions(label_distributions, select_labels))
    negative_labels.append(find_max_data_cluster_distributions(label_distributions, select_labels))
    norm_label_distributions = norm_cluster_label_distributions(label_distributions)
    # norm_label_distributions = norm_cluster_label_distributions(label_distributions)
    # neg_first = find_max_label_distance(positive_labels[0],norm_label_distributions,select_labels)
    # negative_labels.append(select_labels.pop(neg_first))
    # 两者同时开始寻找离自己最近的label_distributions（欧氏距离），如果找到的不是同一个，那么不需另做处理，
    # 如果找到的是同一个，则计算两者中心的欧式距离，谁近归谁，没搜寻到的则去剩下的里面找一个欧式距离最近的
    while select_labels:
        min_pos_index = find_min_label_distance(positive_labels, norm_label_distributions, select_labels)
        min_neg_index = find_min_label_distance(negative_labels, norm_label_distributions, select_labels)
        if min_neg_index == min_pos_index:
            same_min_index = min_pos_index
            if find_min_space_distance(positive_labels,negative_labels, cluster_center, select_labels[same_min_index]):
                positive_labels.append(select_labels.pop(same_min_index))
                if select_labels:
                    min_neg_index = find_min_label_distance(negative_labels, norm_label_distributions, select_labels)
                    negative_labels.append(select_labels.pop(min_neg_index))
                else:
                    break
            else:
                negative_labels.append(select_labels.pop(same_min_index))
                if select_labels:
                    min_pos_index = find_min_label_distance(positive_labels, norm_label_distributions, select_labels)
                    positive_labels.append(select_labels.pop(min_pos_index))
                else:
                    break
        else:
            if min_pos_index > min_neg_index:
                positive_labels.append(select_labels.pop(min_pos_index))
                negative_labels.append(select_labels.pop(min_neg_index))
            else:
                negative_labels.append(select_labels.pop(min_neg_index))
                positive_labels.append(select_labels.pop(min_pos_index))
    # print(positive_labels)
    # print(negative_labels)
    labels = change_pos_neg_label_to_labels(positive_labels, negative_labels, label_distributions.shape[0])
    # label_distributions
    return labels


# 返回的是第i个簇 同时select_labels中的这个值会被取出
def find_max_data_cluster_distributions(label_distributions, select_labels):
    data_max_size = 0
    max_data_index = None
    for i in range(len(select_labels)):
        if label_distributions[select_labels[i]][label_distributions.shape[1]-1] > data_max_size:
            data_max_size = label_distributions[select_labels[i]][label_distributions.shape[1]-1]
            max_data_index = i
    max_cluster_i = select_labels.pop(max_data_index)
    return max_cluster_i


def find_min_label_distance(part_labels, norm_label_distributions, select_labels):
    min_label_distance = float('inf')
    min_cluster_index = None
    for cluster_i in part_labels:
        min_label_distance_i = float('inf')
        min_cluster_index_i = None
        norm_label_distributions_i = norm_label_distributions[cluster_i, :]
        for cluster_j in range(len(select_labels)):
            norm_label_distributions_j = norm_label_distributions[select_labels[cluster_j], :]
            distance_ij = euclidean_distance(norm_label_distributions_i, norm_label_distributions_j)
            if distance_ij < min_label_distance_i:
                min_label_distance_i = distance_ij
                min_cluster_index_i = cluster_j
        if min_label_distance_i < min_label_distance:
            min_label_distance = min_label_distance_i
            min_cluster_index = min_cluster_index_i
    return min_cluster_index


def find_max_label_distance(pos_label, norm_label_distributions, select_labels):
    max_label_distance = 0
    max_cluster_index = None
    for cluster_j in range(len(select_labels)):
        norm_label_distributions_j = norm_label_distributions[select_labels[cluster_j], :]
        norm_label_distributions_i = norm_label_distributions[pos_label, :]
        distance_ij = euclidean_distance(norm_label_distributions_i, norm_label_distributions_j)
        if distance_ij > max_label_distance:
            max_label_distance = distance_ij
            max_cluster_index = cluster_j
    return max_cluster_index


def change_pos_neg_label_to_labels(positive_labels, negative_labels, length):
    labels = np.zeros(length)
    for pos in positive_labels:
        labels[pos] = 1
    for neg in negative_labels:
        labels[neg] = 0
    return labels


# 如果是属于positive 则返回True 属于negative_part则返回false
def find_min_space_distance(positive_labels, negative_labels, cluster_centers, select_labels_i):
    pos_min_space_distance = float('inf')
    neg_min_space_distance = float('inf')
    for pos_label in positive_labels:
        if euclidean_distance(cluster_centers[pos_label], cluster_centers[select_labels_i]) < pos_min_space_distance:
            pos_min_space_distance = euclidean_distance(cluster_centers[pos_label], cluster_centers[select_labels_i])
    for neg_label in negative_labels:
        if euclidean_distance(cluster_centers[neg_label], cluster_centers[select_labels_i]) < neg_min_space_distance:
            neg_min_space_distance = euclidean_distance(cluster_centers[neg_label], cluster_centers[select_labels_i])
    if pos_min_space_distance < neg_min_space_distance:
        return True
    else:
        return False


def norm_cluster_label_distributions(label_distributions):
    norm_label_distributions = []
    for label_distribution_i in label_distributions:
        label_length = len(label_distribution_i)
        cluster_distribution = label_distribution_i[0:label_length - 1]
        cluster_size = label_distribution_i[label_length - 1]
        norm_label_distribution_i = cluster_distribution / cluster_size
        norm_label_distributions.append(norm_label_distribution_i)
    norm_label_distributions = np.array(norm_label_distributions)
    return norm_label_distributions


def cal_combine_list(labels, pure_partial_label):
    label_list = []
    combine_list = []
    pure_partial_label = np.array(pure_partial_label)
    for u in np.unique(labels):
        label_list.append(pure_partial_label[labels == u])
    for tmp_list in label_list:
        combine_list.append(combine_part_list(tmp_list))
    return combine_list


def combine_part_list(label_list):
    combine_list = None
    for tmp in label_list:
        if combine_list is None:
            combine_list = tmp
        else:
            combine_list += tmp
    combine_list[combine_list >=1] = 1
    return combine_list


def clustering(data, k):
    k_means = KMeans(n_clusters=k).fit(data)
    labels = k_means.labels_
    return labels


# def clustering(data, k):
#     k_means = SpectralClustering(n_clusters=k).fit(data)
#     labels = k_means.labels_
#     return labels

# 不同的center纯度设置可能需要不一样
def cal_lowest_purity(label_distributions):
    lowest_purity = 1
    for label_distribution_i in label_distributions:
        label_length = len(label_distribution_i)
        cluster_distribution = label_distribution_i[0:label_length - 1]
        cluster_size = label_distribution_i[label_length - 1]
        cluster_pure = cluster_distribution / cluster_size
        cluster_purity_i = np.max(cluster_pure)
        if lowest_purity > cluster_purity_i:
            lowest_purity = cluster_purity_i
    return lowest_purity


# def cal_pure_label_set(label_distributions):
#     cluster_pure_label_set = []
#     purity = cal_lowest_purity(label_distributions)
#     for label_distribution_i in label_distributions:
#         label_length = len(label_distribution_i)
#         cluster_distribution = label_distribution_i[0:label_length - 1]
#         cluster_size = label_distribution_i[label_length - 1]
#         cluster_pure = cluster_distribution / cluster_size
#         cluster_pure[cluster_pure >= purity] = 1
#         cluster_pure[cluster_pure < purity] = 0
#         cluster_pure_label_set.append(cluster_pure)
#     cluster_pure_label_set = np.array(cluster_pure_label_set)
#     return cluster_pure_label_set


def cal_pure_label_set(label_distributions):
    cluster_pure_label_set =[]
    for label_distribution_i in label_distributions:
        label_length = len(label_distribution_i)
        cluster_distribution = label_distribution_i[0:label_length - 1]
        cluster_pure = np.sum(cluster_distribution)/len(cluster_distribution)
        cluster_pure_set = np.zeros(len(cluster_distribution))
        cluster_pure_set[cluster_distribution > cluster_pure] = 1
        cluster_pure_label_set.append(cluster_pure_set)
    cluster_pure_label_set = np.array(cluster_pure_label_set)
    return cluster_pure_label_set


def cal_small_class_label_set(label_distributions):
    small_class_label_set = []
    all_label_distributions = np.sum(label_distributions, axis=0)
    label_length = len(all_label_distributions)
    distribution = all_label_distributions[0:label_length - 1]
    data_size = all_label_distributions[label_length - 1]
    label_size = np.sum(distribution)
    partial_level = data_size/label_size
    partial_distribution = all_label_distributions/len(distribution)/partial_level
    for i in range(label_distributions.shape[0]):
        tmp_distribution = label_distributions[i, 0:label_length - 1]
        small_class_distribution = np.zeros(tmp_distribution.shape)
        for j in range(len(tmp_distribution)):
            if tmp_distribution[j] > partial_distribution[j]:
                small_class_distribution[j] = 1

        small_class_label_set.append(small_class_distribution)
    small_class_label_set = np.array(small_class_label_set)
    return small_class_label_set


def combine_pure_and_small_class_label_set(pure_label_set, small_class_label_set):
    final_label_set = pure_label_set + small_class_label_set
    final_label_set[final_label_set > 1] = 1
    return final_label_set


def cal_label_num(label_distributions, label_i):
    return np.sum(label_distributions[:,label_i])


def distinguish_share_or_interfere_label(label_distributions, final_label_set, labels, combine_label_set):
    same_label_list = []
    combine_label_set = np.array(combine_label_set)
    for i in range(final_label_set.shape[1]):
        if combine_label_set[0][i] == 1 and combine_label_set[1][i] == 1:
            same_label_list.append(i)
    mask = []
    for tmp in np.unique(labels):
        mask.append(labels == tmp)
    tmp_distributions = []
    tmp_label_set = []
    for tmp_mask in mask:
        tmp_distributions.append(label_distributions[tmp_mask])
        tmp_label_set.append(final_label_set[tmp_mask])
    # print(tmp_distributions)
    for s_l in same_label_list:
        tmp_same_distributions = []
        for i in range(len(mask)):
            tmp_set_i = tmp_label_set[i]
            tmp_distribution_i = tmp_distributions[i]
            tmp_same_distributions.append(tmp_distribution_i[tmp_set_i[:, s_l] == 1])
        combine_label_set = deal_same_label_set(tmp_same_distributions, combine_label_set, s_l)
        # positive_num_i = cal_label_num(tmp_same_distributions[0], s_l)
        # negative_num_i = cal_label_num(tmp_same_distributions[1], s_l)
        # # if positive_num_i/negative_num_i > 11/9:
        # if positive_num_i / negative_num_i >= 0.5:
        #     combine_label_set[1][s_l] = 0
        # # elif positive_num_i/negative_num_i < 9/11:
        # elif positive_num_i / negative_num_i < 0.5:
        #     combine_label_set[0][s_l] = 0
    combine_label_set = cal_small_label_distribution(combine_label_set, tmp_distributions)
    # 要在之前就将有都1部分和都0部分挑出来分别丢进来，不然这样会变成有share标签，之前补充小类样本会比之后补充更有效
    return combine_label_set


def deal_same_label_set(same_distributions, combine_label_set, same_i):
    pos_same_label_i_num = cal_label_num(same_distributions[0], same_i)
    pos_cluster_sum_i = cal_label_num(same_distributions[0], same_distributions[0].shape[1]-1)
    neg_same_label_i_num = cal_label_num(same_distributions[1], same_i)
    neg_cluster_sum_i = cal_label_num(same_distributions[1], same_distributions[1].shape[1]-1)
    if pos_cluster_sum_i == 0:
        combine_label_set[0][same_i] = 0
    elif neg_cluster_sum_i == 0:
        combine_label_set[1][same_i] = 0
    elif pos_same_label_i_num/pos_cluster_sum_i > neg_same_label_i_num/neg_cluster_sum_i:
        combine_label_set[1][same_i] = 0
    else:
        combine_label_set[0][same_i] = 0
    return combine_label_set


def deal_same_label_set_try(same_distributions, combine_label_set, same_i):
    pos_same_label_i_num = cal_label_num(same_distributions[0], same_i)
    pos_cluster_sum_i = cal_label_num(same_distributions[0], same_distributions[0].shape[1]-1)
    neg_same_label_i_num = cal_label_num(same_distributions[1], same_i)
    neg_cluster_sum_i = cal_label_num(same_distributions[1], same_distributions[1].shape[1]-1)
    if pos_cluster_sum_i == 0:
        combine_label_set[0][same_i] = 0
    elif neg_cluster_sum_i == 0:
        combine_label_set[1][same_i] = 0
    elif pos_same_label_i_num/pos_cluster_sum_i*pos_cluster_sum_i > neg_same_label_i_num/neg_cluster_sum_i*neg_cluster_sum_i:
        combine_label_set[1][same_i] = 0
    else:
        combine_label_set[0][same_i] = 0
    return combine_label_set


# 传进来的combine_label_set是未经处理之前的set
def cal_small_label_distribution(combine_label_set, tmp_distributions):
    small_label_set = []
    for i in range(combine_label_set.shape[1]):
        if combine_label_set[0][i] == 0 and combine_label_set[1][i] == 0:
            small_label_set.append(i)
    for small_i in small_label_set:
        positive_small_num = cal_label_num(tmp_distributions[0], small_i)
        negative_small_num = cal_label_num(tmp_distributions[1], small_i)
        judge_factor = judge_small_label(positive_small_num, negative_small_num)
        if judge_factor:
            if positive_small_num/(positive_small_num+negative_small_num) > 0.5:
                combine_label_set[0][small_i] = 1
            else:
                combine_label_set[1][small_i] = 1
    return combine_label_set


def judge_small_label(positive_num, negative_num):
    small_label_sum = positive_num+negative_num
    if small_label_sum == 0:
        # print("have no label data")
        return False
    elif positive_num/small_label_sum >= 0.5or negative_num/small_label_sum >= 0.5:
        return True
    else:
        # print("same")
        return False


def euclidean_distance(x, y, weights=None):
    x = np.array(x).reshape(1, -1)
    y = np.array(y).reshape(1, -1)
    assert len(x) == len(y)
    if weights is None:
        weights = np.ones(len(x))
    distance = np.sqrt(np.sum(np.power(x - y, 2)*weights))
    return distance


def change_cluster_to_classify(cluster_data, separate_labels):
    # print(np.unique(separate_labels))
    mask = separate_labels == 0
    # print(mask)
    train_data = None
    binary_label = None
    positive_data = []
    negative_data = []
    for mask_i, cluster_i in zip(mask, cluster_data):
        if mask_i:
            positive_data.append(cluster_i)
        else:
            negative_data.append(cluster_i)
    for cluster_i in positive_data:
        if train_data is None:
            train_data = cluster_i
            binary_label = np.full(train_data.shape[0], 1)
        else:
            train_data = np.vstack((train_data, cluster_i))
            binary_label = np.hstack((binary_label, np.full(cluster_i.shape[0], 1)))
    for cluster_i in negative_data:
        if train_data is None:
            train_data = cluster_i
            binary_label = np.full(train_data.shape[0], -1)
        else:
            train_data = np.vstack((train_data, cluster_i))
            binary_label = np.hstack((binary_label, np.full(cluster_i.shape[0], -1)))
    return train_data, binary_label


# if __name__ == "__main__":
#     base_path = "mat"
#     real_data_set = "BirdSong.mat"
#     path = os.path.join(base_path, 'BirdSong.mat')
#     test_data, test_label, test_partial_label = Uci_to_mat.load_data(path)
#     part_disambiguate(test_partial_label)
    # s_folder = StratifiedKFold(n_splits=5)
    # for train_index, test_index in s_folder.split(test_data, test_label):
    #     data_train, data_test = test_data[train_index],test_data[test_index]
    #     label_train, label_test = test_label[train_index],test_label[test_index]
    #     partial_label_train, partial_label_test = test_partial_label[train_index], test_partial_label[test_index]
    # encoder(test_data, test_partial_label)