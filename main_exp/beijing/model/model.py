# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class NodeRPEModule(nn.Module):

    def __init__(self, config):
        super(NodeRPEModule, self).__init__()

        self.config = config
        self.dist_embedding = nn.Linear(1, self.config.rpe_dims)
        self.sim_embedding = nn.Linear(1, self.config.rpe_dims)
        self.merge_projection = nn.Linear(self.config.rpe_dims*2, self.config.hidden_dims)

    def forward(self, dist_matrix, sim_matrix):

        N = dist_matrix.size(0)

        dist_features = (dist_matrix.sum(dim=1, keepdim=True) / N).to(torch.float32)
        sim_features = (sim_matrix.sum(dim=1, keepdim=True) / N).to(torch.float32)

        dist_embeddings = self.dist_embedding(dist_features)
        sim_embeddings = self.sim_embedding(sim_features)

        combined_embeddings = torch.cat([dist_embeddings, sim_embeddings], dim=1)

        rpe = self.merge_projection(combined_embeddings)

        return rpe

class GraphTransformerLayer(nn.Module):
    
    def __init__(self, hidden_dims, head_num):
        super(GraphTransformerLayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(hidden_dims, head_num, batch_first=True)
        self.linear1 = nn.Linear(hidden_dims, hidden_dims)
        self.linear2 = nn.Linear(hidden_dims, hidden_dims)
        self.norm1 = nn.LayerNorm(hidden_dims)
        self.norm2 = nn.LayerNorm(hidden_dims)

    def forward(self, x, adj, attn_mask=None):
        
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_output)
        ff_output = self.linear2(F.relu(self.linear1(x)))
        x = self.norm2(x + ff_output)
        return x

class MultiheadAttentionWithSoftAdjacency(nn.Module):

    def __init__(self, hidden_dims, head_num, alpha=0.5):
        super(MultiheadAttentionWithSoftAdjacency, self).__init__()
        self.head_num = head_num
        self.hidden_dims = hidden_dims
        self.alpha = alpha

        self.query_projection = nn.Linear(hidden_dims, hidden_dims * head_num)
        self.key_projection = nn.Linear(hidden_dims, hidden_dims * head_num)
        self.value_projection = nn.Linear(hidden_dims, hidden_dims * head_num)

        self.output_projection = nn.Linear(hidden_dims * head_num, hidden_dims)

        self.norm1 = nn.LayerNorm(hidden_dims)
        self.norm2 = nn.LayerNorm(hidden_dims)
        self.linear1 = nn.Linear(hidden_dims, hidden_dims)
        self.linear2 = nn.Linear(hidden_dims, hidden_dims)

    def forward(self, x, adj_matrix):
        batch_size, seq_len, _ = x.size()

        queries = self.query_projection(x).view(batch_size, seq_len, self.head_num, self.hidden_dims)
        keys = self.key_projection(x).view(batch_size, seq_len, self.head_num, self.hidden_dims)
        values = self.value_projection(x).view(batch_size, seq_len, self.head_num, self.hidden_dims)

        attention_scores = torch.einsum("bqhd,bkhd->bhqk", queries, keys) / torch.sqrt(torch.tensor(self.hidden_dims, dtype=torch.float32))

        adj = adj_matrix.float()
        soft_mask = self.alpha * adj + (1 - self.alpha)
        attention_scores = attention_scores * soft_mask

        attention_weights = F.softmax(attention_scores, dim=-1)

        attn_output = torch.einsum("bhqk,bkhd->bqhd", attention_weights, values).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1) 

        attn_output = self.output_projection(attn_output)

        x = self.norm1(x + attn_output)

        ff_output = self.linear2(F.relu(self.linear1(x)))

        x = self.norm2(x + ff_output)

        return x, attention_weights

class SoftAdaptiveLocalityPooling(nn.Module):
    def __init__(self, hidden_dims, locality_num):
        super(SoftAdaptiveLocalityPooling, self).__init__()
        self.locality_num = locality_num
        self.locality_centers = nn.Parameter(torch.randn(size=(locality_num, hidden_dims)))
        self.linear_q = nn.Linear(hidden_dims, hidden_dims)
        self.linear_k = nn.Linear(hidden_dims, hidden_dims)

    def forward(self, segment_features):
        queries = self.linear_q(self.locality_centers)
        keys = self.linear_k(segment_features)
        values = segment_features

        attention_scores = torch.matmul(queries, keys.T) / (self.locality_centers.size(1) ** 0.5)
        segment2locality_assignments = F.softmax(attention_scores, dim=0)

        locality_features = torch.matmul(segment2locality_assignments, values) + self.locality_centers # [locality_num, hidden_dims]

        return locality_features, segment2locality_assignments

class SoftAdaptiveRegionPooling(nn.Module):
    def __init__(self, hidden_dims, region_num):
        super(SoftAdaptiveRegionPooling, self).__init__()
        self.region_num = region_num
        self.region_centers = nn.Parameter(torch.randn(size=(region_num, hidden_dims)))
        self.linear_q = nn.Linear(hidden_dims, hidden_dims)
        self.linear_k = nn.Linear(hidden_dims, hidden_dims)

    def forward(self, locality_features):
        queries = self.linear_q(self.region_centers)
        keys = self.linear_k(locality_features)
        values = locality_features

        attention_scores = torch.matmul(queries, keys.T) / (self.region_centers.size(1) ** 0.5)
        locality2region_assignments = F.softmax(attention_scores, dim=0)

        region_features = torch.matmul(locality2region_assignments, values) + self.region_centers

        return region_features, locality2region_assignments

class HiFiRoad(nn.Module):

    def __init__(self, config):
        super(HiFiRoad, self).__init__()
        self.config = config

        self.feature_embedding_module = FeatureEmbeddingModule(config)

        self.rpe_module = NodeRPEModule(config)

        self.hierarchical_gnn_module = HierarchicalGNNModule(config)

        self.multi_frequency_module = MultiFrequencyModule(config)

    def forward(self, segment_features, segment_adj_matrix, dist_matrix, sim_matrix):

        init_segment_features = self.feature_embedding_module(segment_features)
        rpe = self.rpe_module(dist_matrix, sim_matrix)

        init_segment_features, updated_segment_features, updated_locality_features, updated_region_features, segment2locality_assignments, locality2region_assignments = \
            self.hierarchical_gnn_module(init_segment_features, segment_adj_matrix, rpe)

        reconstructed_segment_features, high_freq_features = self.multi_frequency_module(init_segment_features, updated_segment_features, segment_adj_matrix)

        return init_segment_features, high_freq_features, reconstructed_segment_features, updated_segment_features, updated_locality_features, updated_region_features, \
            segment2locality_assignments, locality2region_assignments

class FeatureEmbeddingModule(nn.Module):

    def __init__(self, config):
        super(FeatureEmbeddingModule, self).__init__()
        
        self.config = config

        self.segment_lane_emb_layer = nn.Embedding(
            self.config.segment_lane_num, self.config.segment_lane_dims
        )
        self.segment_type_emb_layer = nn.Embedding(
            self.config.segment_type_num, self.config.segment_type_dims
        )
        self.segment_length_emb_layer = nn.Embedding(
            self.config.segment_length_num, self.config.segment_length_dims
        )
        self.segment_id_emb_layer = nn.Embedding(
            self.config.segment_id_num, self.config.segment_id_dims
        )

        self.emb_dims = self.config.segment_id_dims + self.config.segment_lane_dims + \
            self.config.segment_type_dims + self.config.segment_length_dims
        
        self.emb_linear = nn.Linear(
            self.emb_dims, self.config.hidden_dims
        )

    def forward(self, segment_features):
        
        segment_lane_emb = self.segment_lane_emb_layer(segment_features[:, 0])
        segment_type_emb = self.segment_type_emb_layer(segment_features[:, 1])
        segment_length_emb = self.segment_length_emb_layer(segment_features[:, 2])
        segment_id_emb = self.segment_id_emb_layer(segment_features[:, 3])

        segment_emb = torch.cat([
            segment_lane_emb,
            segment_type_emb,
            segment_length_emb,
            segment_id_emb
        ], dim=1)

        init_segment_features = self.emb_linear(segment_emb)

        return init_segment_features

class HierarchicalGNNModule(nn.Module):

    def __init__(self, config):
        super(HierarchicalGNNModule, self).__init__()
        
        self.config = config

        self.locality_pooling = SoftAdaptiveLocalityPooling(self.config.hidden_dims, self.config.locality_num)
        self.region_pooling = SoftAdaptiveRegionPooling(self.config.hidden_dims, self.config.region_num)
        
        self.region_transformer = GraphTransformerLayer(self.config.hidden_dims, self.config.head_num)
        self.locality_transformer = GraphTransformerLayer(self.config.hidden_dims, self.config.head_num)
        self.segment_transformer = GraphTransformerLayer(self.config.hidden_dims, self.config.head_num)

        self.locality_update_gate = nn.Linear(self.config.hidden_dims*2, 1)
        self.segment_update_gate = nn.Linear(self.config.hidden_dims*2, 1)

    def forward(self, init_segment_features, segment_adj_matrix, rpe):
        
        segment_features = init_segment_features + rpe

        locality_features, segment2locality_assignments = self.locality_pooling(segment_features)
        locality_adj_matrix = segment2locality_assignments @ segment_adj_matrix @ segment2locality_assignments.T
        
        region_features, locality2region_assignments = self.region_pooling(locality_features)
        region_adj_matrix = locality2region_assignments @ locality_adj_matrix @ locality2region_assignments.T

        updated_region_features = self.region_transformer(region_features.unsqueeze(0), region_adj_matrix).squeeze(0)

        locality_updates = locality2region_assignments.T @ updated_region_features  # [locality_num, hidden_dims]
        locality_update_ratio = F.sigmoid(self.locality_update_gate(torch.cat((locality_features, locality_updates), dim=-1)))
        locality_features = (1 - locality_update_ratio) * locality_features + locality_update_ratio * locality_updates
        updated_locality_features = self.locality_transformer(locality_features.unsqueeze(0), locality_adj_matrix).squeeze(0)

        segment_updates = segment2locality_assignments.T @ updated_locality_features
        segment_update_ratio = F.sigmoid(self.segment_update_gate(torch.cat((segment_features, segment_updates), dim=-1)))
        segment_features = (1 - segment_update_ratio) * segment_features + segment_update_ratio * segment_updates
        updated_segment_features = self.segment_transformer(segment_features.unsqueeze(0), segment_adj_matrix).squeeze(0)

        return init_segment_features, updated_segment_features, updated_locality_features, updated_region_features, segment2locality_assignments, locality2region_assignments
    
class MultiFrequencyModule(nn.Module):
    
    def __init__(self, config):
        super(MultiFrequencyModule, self).__init__()
        
        self.config = config

        self.alpha = nn.Parameter(torch.ones(1))

        self.low_freq_module = GraphTransformerLayer(self.config.hidden_dims, self.config.head_num)
        self.high_freq_module = GraphTransformerLayer(self.config.hidden_dims, self.config.head_num)

    def forward(self, init_features, low_freq_features, segment_adj_matrix):
        
        high_freq_features = init_features - low_freq_features

        updated_low_freq_features = self.low_freq_module(low_freq_features.unsqueeze(0), segment_adj_matrix).squeeze(0)
        updated_high_freq_features = self.high_freq_module(high_freq_features.unsqueeze(0), segment_adj_matrix).squeeze(0)

        reconstructed_features = (1 - self.alpha) * updated_low_freq_features + (self.alpha) * updated_high_freq_features

        return reconstructed_features, high_freq_features