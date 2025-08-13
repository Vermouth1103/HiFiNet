import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import pickle
from utils import *
from model.label_prediction_model import *
import numpy as np
import random
import argparse
import time
import logging
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def preset_log(args):

    task = "label_prediction"
    log_dir = f"./main_exp/{args.city}/log/{task}/"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        filename = os.path.join(log_dir, f"experiment_{args.timestamp}.log"),
        filemode = "w",
        level = logging.INFO,
        format = "%(asctime)s - %(levelname)s - %(message)s"
    )

def evaluate(model, test_segments, test_labels):
    
    model.eval()
    with torch.no_grad():
        test_probs = model(test_segments).cpu().numpy().squeeze()
    
    test_preds = (test_probs >= 0.5).astype(int)
    accuracy = accuracy_score(test_labels, test_preds)
    precision = precision_score(test_labels, test_preds, zero_division=0)
    recall = recall_score(test_labels, test_preds, zero_division=0)
    f1 = f1_score(test_labels, test_preds, zero_division=0)
    auc = roc_auc_score(test_labels, test_probs)

    return accuracy, precision, recall, f1, auc

def train_label_prediction(args):

    preset_log(args)

    with open(args.config, "r") as f:
        config = json.load(f)
    config = dict_to_object(config)
    config.seed = args.seed
    config.city = args.city

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.device)
    config.device = torch.device(
        f"cuda:{config.device}" if torch.cuda.is_available() else "cpu")
    print(config)
    logging.info(f"Experiment Config: {config}")

    segment_features, segment_adj_matrix, segment2locality_assignments, locality2region_assignments, segment_dist_matrix, segment_sim_matrix = \
        load_downstream_task_data(config)
    print(segment_features.shape, np.array(segment_adj_matrix).shape, np.array(segment2locality_assignments).shape, np.array(locality2region_assignments).shape, np.array(segment_dist_matrix).shape, np.array(segment_sim_matrix).shape)

    segment_features = torch.tensor(segment_features.astype(int), dtype=torch.long).to(config.device)
    segment_adj_matrix = torch.tensor(segment_adj_matrix, dtype=torch.float).to(config.device)
    segment2locality_assignments = torch.tensor(
        segment2locality_assignments, dtype=torch.float).to(config.device)
    locality2region_assignments = torch.tensor(
        locality2region_assignments, dtype=torch.float).to(config.device)
    segment_dist_matrix = torch.tensor(segment_dist_matrix).to(config.device)
    segment_sim_matrix = torch.tensor(segment_sim_matrix).to(config.device)

    model = LabelPredictionModel(config, 
        segment_features, segment_adj_matrix, segment_dist_matrix, segment_sim_matrix, segment2locality_assignments, locality2region_assignments).to(config.device)
    print(model)

    bce_criterion = torch.nn.BCELoss()

    model_optimizer = optim.Adam(
        model.parameters(), lr=config.label_prediction_learning_rate)

    max_f1 = -float("inf")
    max_auc = -float("inf")
    for epoch in tqdm(range(config.label_prediction_epochs)):

        model_optimizer.zero_grad()

        train_segment_set, train_label_set, test_segment_set, test_label_set = load_label_prediction_train_data(config)

        train_segments = torch.tensor(
            train_segment_set, dtype=torch.long).to(config.device)
        train_labels = torch.tensor(
            train_label_set, dtype=torch.float).to(config.device)
        test_segments = torch.tensor(
            test_segment_set, dtype=torch.long).to(config.device)

        train_probs = model(train_segments).squeeze()
        
        loss = bce_criterion(train_probs, train_labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
        model_optimizer.step()

        accuracy, precision, recall, f1, auc = evaluate(model, test_segments, test_label_set)
        logging.info(f"epoch: {epoch} loss: {loss.item()} accuracy: {accuracy} precision: {precision:.3f} recall: {recall:.3f} f1: {f1:.3f} auc: {auc:.3f}")

        if f1>max_f1 or auc>max_auc:
            if f1 > max_f1:
                max_f1 = f1
            if auc > max_auc:
                max_auc = auc

            output_label_prediction_dir = os.path.join(config.output_dir, "label_prediction_model")
            if not os.path.exists(output_label_prediction_dir):
                os.makedirs(output_label_prediction_dir)
            torch.save(model.state_dict(), os.path.join(output_label_prediction_dir, f"label_prediction_model_{args.timestamp}"))

    logging.info(f"max f1: {max_f1} max auc: {max_auc}")
    print(f"max f1: {max_f1} max auc: {max_auc}")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=str, default="beijing")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    args.config = f"./main_exp/{args.city}/config/label_prediction_config.json"

    setup_seed(args.seed)

    timestamp = time.localtime(time.time())
    timestamp_str = time.strftime("%Y-%m-%d_%H-%M-%S", timestamp)
    args.timestamp = timestamp_str

    train_label_prediction(args)
