import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import pickle
from utils import *
from model.route_plan_model import *
import numpy as np
import random
import argparse
import time
import logging
import json
from tqdm import tqdm

def preset_log(args):

    task = "route_plan"
    log_dir = f"./main_exp/{args.city}/log/{task}/"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        filename = os.path.join(log_dir, f"experiment_{args.timestamp}.log"),
        filemode = "w",
        level = logging.INFO,
        format = "%(asctime)s - %(levelname)s - %(message)s"
    )

def temperature_sampling(logits, temperature=1.0):
    logits_adjusted = logits / temperature
    probabilities = F.softmax(logits_adjusted, dim=-1)
    next_token = torch.multinomial(probabilities, num_samples=1)
    return next_token


def top_k_sampling_batch(logits, k=5):
    
    topk_values, topk_indices = torch.topk(logits, k, dim=-1)

    topk_probs = F.softmax(topk_values, dim=-1)

    sampled_idx = torch.multinomial(topk_probs, num_samples=1).squeeze(-1)

    next_token = topk_indices.gather(dim=1, index=sampled_idx.unsqueeze(-1)).squeeze(-1)

    return next_token
    

def evaluate(model, test_route_set, config):

    model.eval()
    pred_right = 0
    recall_right = 0
    pred_sum = 0
    recall_sum = 0
    edt_sum = 0
    edt_count = 0
    pred_list = []

    for batch in test_route_set:

        if len(batch) == 0 or len(batch[0]) < 15:
            continue    
        
        batch = np.array(batch)[:, :15]
        input = batch[:, :6]
        label = batch[:, 6:-1]
        destination = batch[:, -1]
        pred_batch = []
        destination = torch.tensor(destination, dtype=torch.long, device=config.device)
        
        for step in range(batch.shape[1]-7):
            input_tensor = torch.tensor(input, dtype=torch.long, device=config.device)  
            pred = model(input_tensor, destination)    
            pred_loc = torch.argmax(pred, 2).tolist()
            pred_loc = np.array(pred_loc)[:, -1]
            pred_batch.append(pred_loc)
            input = np.concatenate((np.array(input.tolist()), pred_loc[:, np.newaxis]), 1)

        pred_batch = input[:, 6:]
        pred_list.extend(pred_batch.reshape(-1).tolist())

        for index, (tra_pred, tra_label) in enumerate(zip(pred_batch.tolist(), label.tolist())):
            edt_sum += edit(tra_pred, tra_label)     
            edt_count += (len(tra_pred) + len(tra_label)) / 2
            for item in tra_pred:
                if item in tra_label:
                    pred_right += 1
            for item in tra_label:
                if item in tra_pred:
                    recall_right += 1          
            pred_sum += len(tra_pred)
            recall_sum += len(tra_label)            

    print("pred_right:", pred_right, "recall_right:", recall_right, "pred_sum:", pred_sum, "recall_sum:", recall_sum)
    print("pred num:", len(set(list(pred_list))))
    edt = edt_sum / edt_count * 10
    precision = float(pred_right) / pred_sum
    recall = float(recall_right) / recall_sum
    f1 = (2 * precision * recall) / (precision + recall + 1e-9)

    return precision, recall, f1, edt

def train_route_plan(args):

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
    
    segment_features, segment_adj_matrix, segment2locality_assign, locality2region_assign, segment_dist_matrix, segment_sim_matrix = \
        load_downstream_task_data(config)
    print(segment_features.shape, np.array(segment_adj_matrix).shape, np.array(segment2locality_assign).shape, np.array(locality2region_assign).shape)

    train_route_set, test_route_set = load_route_plan_data(config)
    print(len(train_route_set))
    print(len(test_route_set))

    segment_features = torch.tensor(segment_features.astype(int), dtype=torch.long).to(config.device)
    segment_adj_matrix = torch.tensor(segment_adj_matrix, dtype=torch.float).to(config.device)    
    segment2locality_assign = torch.tensor(
        segment2locality_assign, dtype=torch.float).to(config.device)
    locality2region_assign = torch.tensor(
        locality2region_assign, dtype=torch.float).to(config.device)
    segment_dist_matrix = torch.tensor(segment_dist_matrix, dtype=torch.float).to(config.device)
    segment_sim_matrix = torch.tensor(segment_sim_matrix, dtype=torch.float).to(config.device)

    model = RoutePlanModel(config, 
        segment_features, segment_adj_matrix, segment_dist_matrix, segment_sim_matrix, segment2locality_assign, locality2region_assign).to(config.device)
    print(model)

    ce_criterion = torch.nn.CrossEntropyLoss()

    model_optimizer = optim.Adam(
        model.parameters(), lr=config.route_plan_learning_rate)

    max_f1 = -float("inf")
    min_edt = float("inf")
    for epoch in tqdm(range(config.route_plan_epochs)):
        
        for batch in train_route_set:
            
            if len(batch) == 0 or len(batch[0]) < 4: 
                continue     
            
            model.train()
            model_optimizer.zero_grad()

            input = torch.tensor(np.array(batch)[:, :-1], dtype=torch.long, device=config.device)  
            if len(batch[0]) > 10:
                destination = torch.tensor(np.array(batch)[:, 10], dtype=torch.long, device=config.device)
            else:
                destination = torch.tensor(np.array(batch)[:, -1], dtype=torch.long, device=config.device)
            label = torch.tensor(np.array(batch)[:, 1:], dtype=torch.long, device=config.device)
            
            
            pred = model(input, destination)
            pred = pred.view(pred.shape[1], pred.shape[0], pred.shape[2])

            loss = ce_criterion(pred.reshape(-1, config.segment_num), label.reshape(-1))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
            model_optimizer.step()
        
        precision, recall, f1, edt = evaluate(model, test_route_set, config)  
        if f1 > max_f1 or edt < min_edt:
            if f1 > max_f1:
                max_f1 = f1
            if edt < min_edt:
                min_edt = edt
            
            output_route_plan_dir = os.path.join(config.output_dir, "route_plan_model")
            if not os.path.exists(output_route_plan_dir):
                os.makedirs(output_route_plan_dir)
            torch.save(model.state_dict(), os.path.join(output_route_plan_dir, f"route_plan_model_{args.timestamp}"))

        logging.info(f"Route Plan Epoch: {epoch} Loss: {loss.item()} Precision: {precision:.3f} Recall: {recall:.3f} F1: {f1:.3f} EDT: {edt:.3f} max_F1: {max_f1:.3f} min_EDT: {min_edt:.3f}")
        print(f"Route Plan Epoch: {epoch} Loss: {loss.item()} Precision: {precision:.3f} Recall: {recall:.3f} F1: {f1:.3f} EDT: {edt:.3f} max_F1: {max_f1:.3f} min_EDT: {min_edt:.3f}")

    logging.info(f"max_F1: {max_f1:.3f} min_EDT: {min_edt:.3f}")
    print(f"max_F1: {max_f1:.3f} min_EDT: {min_edt:.3f}")

def setup_seed(seed):
    torch.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=str, default="xian")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    args.config = f"./main_exp/{args.city}/config/route_plan_config.json"

    setup_seed(args.seed)
    
    timestamp = time.localtime(time.time())
    timestamp_str = time.strftime("%Y-%m-%d_%H-%M-%S", timestamp)
    args.timestamp = timestamp_str

    train_route_plan(args)

