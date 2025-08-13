from os import sep
import pickle
import numpy as np
from scipy import sparse
import random
import copy
from sklearn.utils import shuffle

EPS = 1e-30

class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

def dict_to_object(dict_obj):
    if not isinstance(dict_obj, dict):
        return dict_obj
    inst = Dict()
    for k, v in dict_obj.items():
        inst[k] = dict_to_object(v)
    return inst

def load_hierarchy_construct_data(config):

    with open(config.segment_features, "rb") as f:
        segment_features = pickle.load(f)

    with open(config.segment_adj, "rb") as f:
        segment_adj = np.array(pickle.load(f))[:config.segment_num, :config.segment_num]

    with open(config.segment_trajectory_adj, "rb") as f:
        segment_trajectory_adj = np.array(pickle.load(f))[:config.segment_num, :config.segment_num]

    with open(config.segment_dist_matrix, "rb") as f:
        segment_dist_matrix = pickle.load(f)
    
    with open(config.segment_sim_matrix, "rb") as f:
        segment_sim_matrix = pickle.load(f)

    with open(config.segment2locality_spectral_labels, "rb") as f:
        segment2locality_spectral_labels = pickle.load(f)

    with open(config.locality2region_spectral_labels, "rb") as f:
        locality2region_spectral_labels = pickle.load(f)

    return segment_features, segment_adj, segment_trajectory_adj, segment_dist_matrix, segment_sim_matrix, \
            segment2locality_spectral_labels, locality2region_spectral_labels

def calculate_cosine_similarity(input1, input2):
    dot_product = input1 @ input2.T

    input1_norm = input1.norm(dim=1, keepdim=True)
    input2_norm = input2.norm(dim=1, keepdim=True)

    norm_product = input1_norm @ input2_norm.T + EPS

    cosine_similarity = dot_product / norm_product

    return cosine_similarity

def load_downstream_task_data(config):

    with open(config.segment_features, "rb") as f:
        segment_features = np.array(pickle.load(f))

    with open(config.segment_adj, "rb") as f:
        segment_adj_matrix = np.array(pickle.load(f))[:config.segment_num, :config.segment_num]
    
    with open(config.segment2locality_assign, "rb") as f:
        segment2locality_assign = np.array(pickle.load(f))[:config.segment_num, :]
    
    with open(config.locality2region_assign, "rb") as f:
        locality2region_assign = np.array(pickle.load(f))

    with open(config.segment_dist_matrix, "rb") as f:
        segment_dist_matrix = pickle.load(f)
    
    with open(config.segment_sim_matrix, "rb") as f:
        segment_sim_matrix = pickle.load(f)

    return segment_features, segment_adj_matrix, segment2locality_assign, locality2region_assign, segment_dist_matrix, segment_sim_matrix

def load_label_prediction_train_data(config):
    
    label_pred_train = pickle.load(open(config.label_train_set, "rb"))
    label_train_true = label_pred_train[:100]
    label_train_false = [] 
    while len(label_train_false) < len(label_train_true):
        node = random.randint(0, config.segment_num - 1)      
        if (not node in label_train_false) and (not node in label_train_true):
            label_train_false.append(node)           

    label_train_set = label_train_false
    label_train_set.extend(label_train_true)
    label_train_real = [0 for i in range(int(len(label_train_set) // 2))]
    label_train_real.extend([1 for i in range(int(len(label_train_set) // 2))])

    label_train_set, label_train_real = shuffle(label_train_set, label_train_real, random_state=config.seed)

    label_test_false = pickle.load(open(config.label_train_set_false, "rb"))
    label_test_true = label_pred_train[100:]
    label_test_set = label_test_false
    label_test_set.extend(label_test_true)
    label_test_real = [0 for i in range(int(len(label_test_set) // 2))]
    label_test_real.extend([1 for i in range(int(len(label_test_set) // 2))]) 

    label_test_set, label_test_real = shuffle(label_test_set, label_test_real, random_state=config.seed)

    return label_train_set, label_train_real, label_test_set, label_test_real

def load_next_location_prediction_data(config):

    with open(config.train_loc_set, "rb") as f:
        train_loc_set = pickle.load(f)
    
    with open(config.test_loc_set, "rb") as f:
        test_loc_set = pickle.load(f)

    return train_loc_set, test_loc_set

def load_destination_prediction_data(config):

    with open(config.train_des_set, "rb") as f:
        train_des_set = pickle.load(f)

    with open(config.test_des_set, "rb") as f:
        test_des_set = pickle.load(f)

    return train_des_set, test_des_set

def load_route_plan_data(config):

    with open(config.train_route_set, "rb") as f:
        train_route_set = pickle.load(f)

    with open(config.test_route_set, "rb") as f:
        test_route_set = pickle.load(f)

    return train_route_set, test_route_set

def edit(str1, str2):
    if len(str1) < len(str2):
        str1, str2 = str2, str1
    
    previous = list(range(len(str2) + 1))
    current = [0] * (len(str2) + 1)
    
    for i in range(1, len(str1) + 1):
        current[0] = i
        for j in range(1, len(str2) + 1):
            d = 0 if str1[i-1] == str2[j-1] else 1
            current[j] = min(previous[j] + 1, current[j-1] + 1, previous[j-1] + d)
        previous, current = current, previous
    
    return previous[len(str2)]
