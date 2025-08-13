import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import pickle
from utils import *
from model.next_location_prediction_model import *
import numpy as np
import random
import argparse
import time
import logging
import json
from tqdm import tqdm

def preset_log(args):

    task = "next_location_prediction"
    log_dir = f"./main_exp/{args.city}/log/{task}/"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        filename = os.path.join(log_dir, f"experiment_{args.timestamp}.log"),
        filemode = "w",
        level = logging.INFO,
        format = "%(asctime)s - %(levelname)s - %(message)s"
    )

def evaluate(model, test_loc_set, config):
    
    model.eval()
    right_1 = 0
    right_5 = 0
    total_num = 0
    
    for batch in test_loc_set:
        
        if len(batch) == 0 or len(batch[0]) < 4:
            continue

        batch = np.array(batch)
        input = batch[:, :-1]
        label = batch[:, -1]
        
        input = torch.tensor(input, dtype=torch.long, device=config.device)  
        label = torch.tensor(label, dtype=torch.long, device=config.device)  

        with torch.no_grad():
            pred = model(input)
        pred_1 = torch.argmax(pred, 2).tolist()
        pred_1 = np.array(pred_1)[:, -1]
        pred_5 = (-np.array(pred.tolist())).argsort()[:, :, :5]
        pred_5 = pred_5[:, -1, :]

        for item1, item2, item3 in zip(pred_1.tolist(), pred_5.tolist(), label.tolist()):
            if item3 == item1:
                right_1 += 1
            if item3 in item2:
                right_5 += 1    
            total_num += 1

    acc1 = float(right_1) / total_num
    acc5 = float(right_5) / total_num
    return acc1, acc5

def train_next_location_prediction(args):

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
    print(segment_features.shape, np.array(segment_adj_matrix).shape, np.array(segment2locality_assignments).shape, np.array(locality2region_assignments).shape)

    segment_features = torch.tensor(segment_features.astype(int), dtype=torch.long).to(config.device)
    segment_adj_matrix = torch.tensor(segment_adj_matrix, dtype=torch.float).to(config.device)
    segment2locality_assignments = torch.tensor(
        segment2locality_assignments, dtype=torch.float).to(config.device)
    locality2region_assignments = torch.tensor(
        locality2region_assignments, dtype=torch.float).to(config.device)
    segment_dist_matrix = torch.tensor(segment_dist_matrix, dtype=torch.float).to(config.device)
    segment_sim_matrix = torch.tensor(segment_sim_matrix, dtype=torch.float).to(config.device)

    model = NextLocationPredictionModel(config, 
        segment_features, segment_adj_matrix, segment_dist_matrix, segment_sim_matrix, segment2locality_assignments, locality2region_assignments).to(config.device)
    print(model)

    train_loc_set, test_loc_set = load_next_location_prediction_data(config)
    print(len(train_loc_set), len(test_loc_set))

    ce_criterion = torch.nn.CrossEntropyLoss()

    model_optimizer = optim.Adam(
        model.parameters(), lr=config.next_location_prediction_learning_rate)

    max_acc1 = -float("inf")
    max_acc5 = -float("inf")
    for epoch in tqdm(range(config.next_location_prediction_epochs)):

        for batch in train_loc_set:
            
            if len(batch) == 0 or len(batch[0]) < 4: 
                continue  

            model.train()
            model_optimizer.zero_grad()  
            
            input = torch.tensor(np.array(batch)[:, :-1], dtype=torch.long, device=config.device)  
            label = torch.tensor(np.array(batch)[:, 1:], dtype=torch.long, device=config.device)  
            
            pred = model(input)

            loss = ce_criterion(pred.reshape(-1, config.segment_num), label.reshape(-1))
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
            model_optimizer.step()

        acc1, acc5 = evaluate(model, test_loc_set, config)
    
        if acc1 > max_acc1 or acc5 > max_acc5:
            if acc1 > max_acc1:
                max_acc1 = acc1
            if acc5 > max_acc5:
                max_acc5 = acc5

            output_model_dir = os.path.join(config.output_dir, "next_location_prediction_model")
            if not os.path.exists(output_model_dir):
                os.makedirs(output_model_dir)
            torch.save(model.state_dict(), os.path.join(output_model_dir, f"next_location_prediction_model_{args.timestamp}"))

        logging.info(f"Next Location Prediction Epoch: {epoch} Loss: {loss.item()} ACC@1: {acc1:.3f} ACC@5: {acc5:.3f} max_ACC@1: {max_acc1:.3f} max_ACC@5: {max_acc5:.3f}")  
        print(f"Next Location Prediction Epoch: {epoch} Loss: {loss.item()} ACC@1: {acc1:.3f} ACC@5: {acc5:.3f} max_ACC@1: {max_acc1:.3f} max_ACC@5: {max_acc5:.3f}")

    logging.info(f"max_ACC@1: {max_acc1:.3f} max_ACC@5: {max_acc5:.3f}")
    print(f"max_ACC@1: {max_acc1:.3f} max_ACC@5: {max_acc5:.3f}")

def setup_seed(seed):
    torch.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=str, default="chengdu")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    args.config = f"./main_exp/{args.city}/config/next_location_prediction_config.json"

    setup_seed(args.seed)
    
    timestamp = time.localtime(time.time())
    timestamp_str = time.strftime("%Y-%m-%d_%H-%M-%S", timestamp)
    args.timestamp = timestamp_str

    train_next_location_prediction(args)
