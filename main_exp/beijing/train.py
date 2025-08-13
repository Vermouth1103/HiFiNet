import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model import *
from utils import *
import os
from tqdm import tqdm
import logging
import numpy as np
import random
import argparse
import json
import time
from scipy.stats import norm

torch.autograd.set_detect_anomaly(True)

def preset_log(args):

    task = 'hierarchy_construction'
    log_dir = f"./main_exp/{args.city}/log/{task}/"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        filename = os.path.join(log_dir, f'experiment_{args.timestamp}.log'),
        filemode = 'w',
        level = logging.INFO,
        format = '%(asctime)s - %(levelname)s - %(message)s'
    )

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def feature_loss(segment_features, locality_features, region_features, segment2locality_assignments, locality2region_assignments, temperature=0.5):
    segment2locality_loss = 0.0
    locality2region_loss = 0.0

    segment2locality = torch.argmax(segment2locality_assignments, dim=0)  
    locality2region = torch.argmax(locality2region_assignments, dim=0)
    unique_localities = torch.unique(segment2locality)
    unique_regions = torch.unique(locality2region)
    segment2locality_sim_matrix = calculate_cosine_similarity(segment_features, locality_features) / temperature
    locality2region_sim_matrix = calculate_cosine_similarity(locality_features, region_features) / temperature

    for locality_idx in unique_localities:
        segment_in_locality = (segment2locality == locality_idx).nonzero(as_tuple=True)[0]
        
        pos_sim = segment2locality_sim_matrix[segment_in_locality, locality_idx]
        pos_pairs = torch.exp(pos_sim)

        neg_sim = segment2locality_sim_matrix[segment_in_locality, :]
        neg_sim[:, locality_idx] = float("-inf")
        neg_pairs = torch.exp(neg_sim).sum(dim=1)

        loss = -torch.log(pos_pairs / (pos_pairs + neg_pairs)).mean()
        segment2locality_loss = segment2locality_loss + loss
    
    segment2locality_loss = segment2locality_loss / len(unique_localities)

    for region_idx in unique_regions:
        locality_in_region = (locality2region == region_idx).nonzero(as_tuple=True)[0]

        pos_sim = locality2region_sim_matrix[locality_in_region, region_idx]
        pos_pairs = torch.exp(pos_sim)

        neg_sim = locality2region_sim_matrix[locality_in_region, :]
        neg_sim[:, region_idx] = float("-inf")
        neg_pairs = torch.exp(neg_sim).sum(dim=1)

        loss = -torch.log(pos_pairs / (pos_pairs + neg_pairs)).mean()
        locality2region_loss = locality2region_loss + loss
    
    locality2region_loss = locality2region_loss / len(unique_regions)

    return segment2locality_loss + locality2region_loss

def assignment_loss(segment2locality_assignments, locality2region_assignments, epsilon=1e-10):
    
    temp = segment2locality_assignments + epsilon
    segment2locality_entropy_loss = -torch.sum(temp * torch.log(temp), dim=1)
    segment2locality_entropy_loss = segment2locality_entropy_loss.mean()

    temp = locality2region_assignments + epsilon
    locality2region_entropy_loss = -torch.sum(temp * torch.log(temp), dim=1)
    locality2region_entropy_loss = locality2region_entropy_loss.mean()

    return segment2locality_entropy_loss + locality2region_entropy_loss

def supervised_assignment_loss(assignments, labels):
    assignments = assignments.transpose(0, 1)
    loss = F.cross_entropy(assignments, labels)
    return loss

def reconstruction_loss(reconstructed_segment_features, segment_adj, threshold=0.5, epsilon=1e-10):
    
    reconstructed_sim_matrix = calculate_cosine_similarity(reconstructed_segment_features, reconstructed_segment_features)

    binary_adj = (segment_adj > 0).float()

    reconstructed_adj = (reconstructed_sim_matrix > threshold).float()

    diff_matrix = torch.abs(reconstructed_adj - binary_adj)

    reconstruction_diff_loss = diff_matrix.mean()

    return reconstruction_diff_loss

def high_freq_loss(init_segment_features, high_freq_features, threshold=0.4, epsilon=1e-10):
    
    init_energy = torch.norm(init_segment_features, p=2, dim=1) ** 2
    high_freq_energy = torch.norm(high_freq_features, p=2, dim=1) ** 2

    energy_ratio = high_freq_energy / (init_energy + epsilon)

    loss = torch.max(energy_ratio - threshold, torch.zeros_like(energy_ratio)).mean()

    return loss

def balance_loss(segment2locality_assignments, locality2region_assignments, epsilon=1e-10):
    
    locality_segment_counts = torch.sum(segment2locality_assignments, dim=1)  # shape [num_localities]
    locality_mean = segment2locality_assignments.shape[1] // segment2locality_assignments.shape[0]  
    locality_balance_loss = torch.mean((locality_segment_counts - locality_mean) ** 2) / segment2locality_assignments.shape[0]

    region_locality_counts = torch.sum(locality2region_assignments, dim=1)  # shape [num_regions]
    region_mean = locality2region_assignments.shape[1] // locality2region_assignments.shape[0]  
    region_balance_loss = torch.mean((region_locality_counts - region_mean) ** 2) / locality2region_assignments.shape[0]

    return locality_balance_loss + region_balance_loss

def trajectory_loss(segment_features, transition_matrix, tau=0.5, neg_sample_size=50):
    
    num_nodes = segment_features.size(0)
    transition_matrix = transition_matrix / torch.max(transition_matrix)

    sim_matrix = calculate_cosine_similarity(segment_features, segment_features)  # shape [num_nodes, num_nodes]
    sim_matrix /= tau 

    pos_mask = transition_matrix > 0 
    pos_weights = transition_matrix[pos_mask]
    pos_sim = sim_matrix[pos_mask]
    weighted_pos_loss = torch.log(torch.sum(pos_weights * torch.exp(pos_sim)))

    neg_indices = []
    for i in range(num_nodes):
        invalid_neg_mask = transition_matrix[i] > 0
        valid_neg_indices = (~invalid_neg_mask).nonzero(as_tuple=True)[0]
        if len(valid_neg_indices) > 0:
            sampled_indices = torch.randperm(len(valid_neg_indices))[:neg_sample_size]
            neg_indices.append(valid_neg_indices[sampled_indices])
        else:
            neg_indices.append(torch.randint(0, num_nodes, (neg_sample_size,)))

    neg_indices = torch.stack(neg_indices)  # shape [num_nodes, neg_sample_size]
    neg_sim = sim_matrix[torch.arange(num_nodes).unsqueeze(1), neg_indices]  # shape [num_nodes, neg_sample_size]
    neg_loss = torch.logsumexp(neg_sim, dim=1).mean()

    loss = -(weighted_pos_loss - neg_loss)
    return loss

def train_hierarchy_construct(args):

    preset_log(args)

    with open(args.config, "r") as f:
        config = json.load(f)
    config = dict_to_object(config)
    config.city = args.city

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.device)
    config.device = torch.device(
        f"cuda:{config.device}" if torch.cuda.is_available() else "cpu")
    print(config)
    logging.info(f"Experiment Config: {config}")

    segment_features, segment_adj_matrix, transition_matrix, segment_dist_matrix, segment_sim_matrix, \
        segment2locality_spectral_labels, locality2region_spectral_labels = load_hierarchy_construct_data(config)
    print(np.array(segment_features).shape, np.array(segment_adj_matrix).shape, np.array(transition_matrix).shape, np.array(segment_dist_matrix).shape, np.array(segment_sim_matrix).shape)

    segment_features = torch.tensor(segment_features).to(config.device)
    segment_adj_matrix = torch.tensor(segment_adj_matrix, dtype=torch.float).to(config.device)
    transition_matrix = torch.tensor(transition_matrix).to(config.device)
    segment_dist_matrix = torch.tensor(segment_dist_matrix).to(config.device)
    segment_sim_matrix = torch.tensor(segment_sim_matrix).to(config.device)
    segment2locality_spectral_labels = torch.tensor(segment2locality_spectral_labels, dtype=torch.long).to(config.device)
    locality2region_spectral_labels = torch.tensor(locality2region_spectral_labels, dtype=torch.long).to(config.device)

    model = HiFiRoad(config).to(config.device)
    print(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    min_loss = float("inf")
    for epoch in tqdm(range(config.epochs)):
        init_segment_features, high_freq_features, reconstructed_segment_features, updated_segment_features, updated_locality_features, updated_region_features, \
            segment2locality_assignments, locality2region_assignments = model(segment_features, segment_adj_matrix, segment_dist_matrix, segment_sim_matrix)
        
        loss_feature = feature_loss(init_segment_features, updated_locality_features, updated_region_features, segment2locality_assignments, locality2region_assignments)
        loss_assignment = supervised_assignment_loss(segment2locality_assignments, segment2locality_spectral_labels) + \
            supervised_assignment_loss(locality2region_assignments, locality2region_spectral_labels)
        loss_reconstruction = reconstruction_loss(reconstructed_segment_features, segment_adj_matrix)
        loss_high_freq = high_freq_loss(init_segment_features, high_freq_features)
        loss_balance = balance_loss(segment2locality_assignments, locality2region_assignments)
        loss_trajectory = trajectory_loss(reconstructed_segment_features, transition_matrix)

        loss = config.loss_feature_ratio * loss_feature + config.loss_assignment_ratio * loss_assignment + \
            config.loss_reconstruction_ratio * loss_reconstruction + config.loss_high_freq_ratio * loss_high_freq + \
            config.loss_balance_ratio * loss_balance + config.loss_trajectory_ratio * loss_trajectory 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logging.info(f"Epoch {epoch + 1}/{config.epochs}, Total Loss: {loss.item():.4f}, \
            Feature Loss: {loss_feature.item():.4f}, Assignment Loss: {loss_assignment.item():.4f}, \
            Reconstruction Loss: {loss_reconstruction.item():.4f}, High_Freq Loss: {loss_high_freq.item():.4f} \
            Balance Loss: {loss_balance.item():.4f}, Trajectory Loss: {loss_trajectory.item():.4f}")

        if loss.item() < min_loss:
            output_assign_dir = os.path.join(config.output_dir, "assign")
            if not os.path.exists(output_assign_dir):
                os.makedirs(output_assign_dir)
            with open(os.path.join(output_assign_dir, f"segment2locality_assign_{args.timestamp}"), "wb") as f:
                pickle.dump(segment2locality_assignments.tolist(), f)
            with open(os.path.join(output_assign_dir, f"locality2region_assign_{args.timestamp}"), "wb") as f:
                pickle.dump(locality2region_assignments.tolist(), f)
            
            output_features_dir = os.path.join(config.output_dir, "features")
            if not os.path.exists(output_features_dir):
                os.makedirs(output_features_dir)
            with open(os.path.join(output_features_dir, f"segment_features_{args.timestamp}"), "wb") as f:
                pickle.dump(reconstructed_segment_features.tolist(), f)
            with open(os.path.join(output_features_dir, f"locality_features_{args.timestamp}"), "wb") as f:
                pickle.dump(updated_locality_features.tolist(), f)
            with open(os.path.join(output_features_dir, f"region_features_{args.timestamp}"), "wb") as f:
                pickle.dump(updated_region_features.tolist(), f)
            with open(os.path.join(output_features_dir, f"high_freq_features_{args.timestamp}"), "wb") as f:
                pickle.dump(high_freq_features.tolist(), f)
            with open(os.path.join(output_features_dir, f"low_frerq_features_{args.timestamp}"), "wb") as f:
                pickle.dump(updated_segment_features.tolist(), f)

            output_model_dir = os.path.join(config.output_dir, "model")
            if not os.path.exists(output_model_dir):
                os.makedirs(output_model_dir)
            torch.save(model.state_dict(), os.path.join(output_model_dir, f"hierarchy_model_{args.timestamp}"))
        
        print(f"Epoch {epoch + 1}/{config.epochs}, Total Loss: {loss.item():.4f}, \
            Feature Loss: {loss_feature.item():.4f}, Assignment Loss: {loss_assignment.item():.4f}, \
            Reconstruction Loss: {loss_reconstruction.item():.4f}, High_Freq Loss: {loss_high_freq.item():.4f} \
            Balance Loss: {loss_balance.item():.4f}, Trajectory Loss: {loss_trajectory.item():.4f}")

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=str, default="beijing")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    args.config = f"./main_exp/{args.city}/config/hierarchy_config.json"

    setup_seed(args.seed)

    timestamp = time.localtime(time.time())
    timestamp_str = time.strftime('%Y-%m-%d_%H-%M-%S', timestamp)
    args.timestamp = timestamp_str

    train_hierarchy_construct(args)