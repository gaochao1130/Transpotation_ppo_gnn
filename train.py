import os
import json
import codecs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import warnings
import random
import argparse

warnings.filterwarnings('ignore')
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import Distribution
from torch.distributions import Categorical

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from torch_geometric.data import Batch
from torch_geometric.nn import GATv2Conv, global_add_pool


# ===================== GNN特征提取器（保持不变） =====================
class PyG_GNN_Extractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128, n_nodes=None,
                 node_feat_dim=None, edge_index=None, edge_attr=None, gnn_hidden=128):
        super().__init__(observation_space, features_dim)
        self.n_nodes = n_nodes
        self.node_feat_dim = node_feat_dim
        self.gnn_hidden = gnn_hidden
        self.register_buffer('edge_index', edge_index)
        if edge_attr is not None:
            self.register_buffer('edge_attr', edge_attr)
        else:
            self.edge_attr = None
        assert observation_space.shape[0] == n_nodes * node_feat_dim, \
            f"观测维度不匹配：期望 {n_nodes * node_feat_dim}，实际 {observation_space.shape[0]}"
        hidden_size = gnn_hidden
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        self.conv1 = GATv2Conv(hidden_size, hidden_size // 4, heads=4, concat=True,
                               edge_dim=1 if edge_attr is not None else None, add_self_loops=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.conv2 = GATv2Conv(hidden_size, hidden_size // 4, heads=4, concat=True,
                               edge_dim=1 if edge_attr is not None else None, add_self_loops=True)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, features_dim),
            nn.LayerNorm(features_dim)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch_size = obs.shape[0]
        device = obs.device
        node_features = obs.view(batch_size, self.n_nodes, self.node_feat_dim)
        x = node_features.view(-1, self.node_feat_dim)
        edge_index = self.edge_index
        offsets = torch.arange(batch_size, device=device) * self.n_nodes
        repeated_edge_index = edge_index.repeat(1, batch_size)
        offsets_per_edge = offsets.repeat_interleave(edge_index.shape[1])
        batch_edge_index = repeated_edge_index + offsets_per_edge.unsqueeze(0)
        if self.edge_attr is not None:
            batch_edge_attr = self.edge_attr.repeat(batch_size, 1)
        else:
            batch_edge_attr = None
        batch = Batch(x=x, edge_index=batch_edge_index, edge_attr=batch_edge_attr)
        batch.batch = torch.arange(batch_size, device=device).repeat_interleave(self.n_nodes)
        valid_mask = batch.x[:, 6].unsqueeze(-1).detach()
        x = self.node_encoder(batch.x)
        x = x * valid_mask
        x_res = x
        x = self.conv1(x, batch.edge_index, batch.edge_attr)
        x = self.norm1(x)
        x = F.elu(x)
        x = x + x_res
        x = x * valid_mask
        x_res = x
        x = self.conv2(x, batch.edge_index, batch.edge_attr)
        x = self.norm2(x)
        x = F.elu(x)
        x = x + x_res
        x = x * valid_mask
        sum_x = global_add_pool(x, batch.batch)
        valid_counts = global_add_pool(valid_mask.squeeze(-1), batch.batch)
        graph_emb = sum_x / (valid_counts.unsqueeze(-1) + 1e-6)
        output = self.output_proj(graph_emb)
        return output


# ===================== 数据加载函数（保持不变） =====================
def load_stations_and_distance():
    with codecs.open("./基础数据/stations.json", "r", encoding="utf-8") as f:
        stations = json.load(f)["stations"]
    dist_matrix = np.loadtxt("./基础数据/distance_matrix.csv", delimiter=",", encoding="utf-8-sig", dtype=np.float32)
    if dist_matrix.shape != (len(stations), len(stations)):
        raise ValueError(f"距离矩阵维度错误：预期 ({len(stations)},{len(stations)})，实际 {dist_matrix.shape}")
    return stations, dist_matrix


def load_supply_demand(stations):
    """加载 20GP 和 40GP 供需数据，返回多箱型张量"""
    import codecs
    import json
    import numpy as np
    
    with codecs.open("./基础数据/supply_data_one.json", "r", encoding="utf-8") as f:
        supply_raw = json.load(f)
    with codecs.open("./基础数据/demand_data_one.json", "r", encoding="utf-8") as f:
        demand_raw = json.load(f)

    supply_20 = np.array(supply_raw["20GPBZ"], dtype=np.float32)
    demand_20 = np.array(demand_raw["20GPBZ"], dtype=np.float32)
    supply_40 = np.array(supply_raw["40GPBZ"], dtype=np.float32)
    demand_40 = np.array(demand_raw["40GPBZ"], dtype=np.float32)

    supply_all = np.stack([supply_20, supply_40], axis=-1)
    demand_all = np.stack([demand_20, demand_40], axis=-1)

    return supply_all, demand_all


# ===================== 熵系数调度回调（保持不变） =====================
class NonLinearEntCoefCallback(BaseCallback):
    def __init__(self, initial_ent_coef=0.1, mid_ent_coef=0.06, final_ent_coef=0.01,
                 total_timesteps=None, mid_ratio=0.3, final_ratio=0.7):
        super().__init__()
        self.initial_ent_coef = initial_ent_coef
        self.mid_ent_coef = mid_ent_coef
        self.final_ent_coef = final_ent_coef
        self.total_timesteps = total_timesteps
        self.mid_ratio = mid_ratio
        self.final_ratio = final_ratio

    def _on_step(self):
        if self.total_timesteps:
            progress = self.num_timesteps / self.total_timesteps
            if progress < self.mid_ratio:
                t = progress / self.mid_ratio
                self.model.ent_coef = self.initial_ent_coef + (self.mid_ent_coef - self.initial_ent_coef) * t
            elif progress < self.final_ratio:
                self.model.ent_coef = self.mid_ent_coef
            else:
                self.model.ent_coef = self.final_ent_coef
        return True


# ===================== 学习率调度函数（保持不变） =====================
def cosine_annealing_lr_with_restart(initial_lr=2e-4, min_lr=3e-5, restart_interval=100000):
    def schedule(progress_remaining):
        total_progress = 1 - progress_remaining
        cycle_progress = (total_progress * 1.5e6) % restart_interval / restart_interval
        lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(cycle_progress * np.pi))
        return lr
    return schedule


# ===================== 环境核心类（带Top-K候选源站） =====================
class ContainerEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, station_names, distance_matrix, max_steps=5000,
                 supply=None, demand=None, scene_list=None, candidate_k=3):
        super().__init__()
        self.station_names = station_names
        self.n_sources = len(station_names)
        self.n_destinations = len(station_names)
        self.n_nodes = self.n_sources
        self.max_steps = max_steps
        self.candidate_k = candidate_k
        self.current_focus_dest = None
        self.dest_partially_served = False

        self.gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cpu_device = torch.device("cpu")
        print(f"环境初始化：重计算使用 {self.gpu_device}，轻量逻辑使用 {self.cpu_device}")

        self.distances = torch.tensor(distance_matrix, dtype=torch.float32, device=self.gpu_device)

        self.scene_list = scene_list if scene_list is not None else []
        self.fixed_scene = (supply is not None and demand is not None)
        if self.fixed_scene:
            self.fixed_supply = supply.copy()
            self.fixed_demand = demand.copy()
        self._next_scene_idx = 0
        self.active_demand_stations = None
        self.box_base = 8

        self.dest_total_mileage = torch.zeros((self.n_destinations,), dtype=torch.float32, device=self.cpu_device)
        # 动作空间：目的站索引 × 候选槽位（每个槽位对应一个候选源站）
        self.action_space = spaces.Discrete(self.n_destinations * self.candidate_k)
        self.completion_step = torch.full((self.n_destinations,), -1.0,
                                          dtype=torch.float32, device=self.cpu_device)
        self.node_feat_dim = 9
        self.observation_dim = self.n_nodes * self.node_feat_dim
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.observation_dim,), dtype=np.float32)
        self.max_possible_dist = torch.max(self.distances).item() + 1e-6

        # 候选集相关
        self.candidates = []          # list of list, 每个目的站长度为 candidate_k
        self.candidate_weights = []   # 可选调试

        self.reset()

    def _choose_scene(self):
        idx = self._next_scene_idx
        self._next_scene_idx = (self._next_scene_idx + 1) % len(self.scene_list)
        supply, demand = self.scene_list[idx]
        return supply.copy(), demand.copy(), idx

    def _update_scene_params(self):
        self.max_supply = float(self.original_supply.max()) if self.original_supply.max() > 0 else 1.0
        self.max_demand = float(self.original_demand.max()) if self.original_demand.max() > 0 else 1.0
        self.total_boxes = int(self.original_demand.sum())
        self.final_base = self.box_base * self.total_boxes * 100

    def _update_candidates(self):
        supply = self.current_supply.cpu().numpy()
        demand = self.current_demand.cpu().numpy()
        dist = self.distances.cpu().numpy()
        K = self.candidate_k
        n_dest = self.n_destinations

        self.candidates = []
        self.candidate_weights = []

        for d in range(n_dest):
            if demand[d].sum() <= 1e-3:
                self.candidates.append([-1] * K)
                self.candidate_weights.append([0.0] * K)
                continue

            valid_src = []
            for s in range(self.n_sources):
                if s != d:
                    A = min(supply[s, 0], demand[d, 0]) + min(supply[s, 1], demand[d, 1])
                    if A > 1e-3:
                        valid_src.append(s)

            if not valid_src:
                self.candidates.append([-1] * K)
                self.candidate_weights.append([0.0] * K)
                continue

            can_fulfill = []
            cannot_fulfill = []
            for s in valid_src:
                A = min(supply[s, 0], demand[d, 0]) + min(supply[s, 1], demand[d, 1])
                D = demand[d].sum()
                if A >= D - 1e-3:
                    can_fulfill.append(s)
                else:
                    cannot_fulfill.append(s)

            can_fulfill.sort(key=lambda s: dist[s, d])
            cannot_fulfill.sort(key=lambda s: (min(supply[s,0], demand[d,0]) + min(supply[s,1], demand[d,1])) / (dist[s, d] + 1e-6), reverse=True)

            candidates_list = can_fulfill + cannot_fulfill
            top_k_src = candidates_list[:K]
            top_k_weights = []
            for s in top_k_src:
                A = min(supply[s, 0], demand[d, 0]) + min(supply[s, 1], demand[d, 1])
                B = dist[s, d] + 1e-6
                top_k_weights.append(A / B)

            while len(top_k_src) < K:
                top_k_src.append(-1)
                top_k_weights.append(0.0)

            self.candidates.append(top_k_src)
            self.candidate_weights.append(top_k_weights)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.fixed_scene:
            supply_np = self.fixed_supply.copy()
            demand_np = self.fixed_demand.copy()
            self.scene_idx = 0
        else:
            supply_np, demand_np, idx = self._choose_scene()
            self.scene_idx = idx
        self.original_supply = torch.tensor(supply_np, dtype=torch.float32, device=self.cpu_device)
        self.original_demand = torch.tensor(demand_np, dtype=torch.float32, device=self.cpu_device)
        self.active_demand_stations = (self.original_demand.sum(dim=-1) > 0).sum().item()
        self.current_supply = self.original_supply.clone()
        self.current_demand = self.original_demand.clone()
        self._update_scene_params()
        self.total_box_mileage = 0.0
        self.step_mileage_sum = 0.0
        self.transport_history = []
        self.steps = 0
        self.allocation_history = []
        self.max_history_len = 20
        self.cumulative_step_reward = 0.0

        self.fulfilled_dest_order = []
        self.fulfilled_dest_dist = []
        self.dest_total_mileage.fill_(0.0)
        self.completion_step = torch.full((self.n_destinations,), -1.0,
                                          dtype=torch.float32, device=self.cpu_device)
        self._update_candidates()
        self.current_focus_dest = None
        self.dest_partially_served = False
        return self._get_state(), {}

    def _get_state(self):
        """
        构建观测状态（CPU端）
        节点特征维度 = 6
        [0] 供应归一化
        [1] 需求归一化
        [2] 需求完成比例
        [3] 有效性标志
        [4] 最近源站距离归一化（由于候选集变化，这里使用候选集中第一个有效源站的距离）
        [5] 专注站指示
        """
        supply_norm = self.current_supply / (self.max_supply + 1e-6)
        demand_norm = self.current_demand / (self.max_demand + 1e-6)
        demand_completed_ratio = 1 - (self.current_demand / (self.original_demand + 1e-6))
        demand_completed_ratio = torch.clamp(demand_completed_ratio, 0.0, 1.0)
        valid_node = ((self.current_supply.sum(dim=-1) > 1e-3) | (self.current_demand.sum(dim=-1) > 1e-3)).float()

        dist_norm = torch.ones(self.n_destinations, device=self.cpu_device)
        dist_cpu = self.distances.cpu()
        for d in range(self.n_destinations):
            if self.candidates[d] and self.candidates[d][0] != -1:
                src = self.candidates[d][0]
                dist_norm[d] = dist_cpu[src, d] / self.max_possible_dist
            else:
                dist_norm[d] = 1.0

        focus_indicator = torch.zeros(self.n_destinations, device=self.cpu_device)
        if self.current_focus_dest is not None:
            focus_indicator[self.current_focus_dest] = 1.0

        node_features = torch.cat([
            supply_norm,                 # 0, 1
            demand_norm,                 # 2, 3
            demand_completed_ratio,      # 4, 5
            valid_node.unsqueeze(1),     # 6
            dist_norm.unsqueeze(1),      # 7
            focus_indicator.unsqueeze(1) # 8
        ], dim=1)

        state_flat = node_features.view(-1)
        return state_flat.cpu().numpy().astype(np.float32)

    def action_masks(self):
        mask = np.zeros(self.n_destinations * self.candidate_k, dtype=bool)
        for d in range(self.n_destinations):
            if self.current_demand[d].sum() <= 1e-3:
                continue
            if self.current_focus_dest is not None and d != self.current_focus_dest:
                continue
            for k, src in enumerate(self.candidates[d]):
                if src != -1:
                    action_idx = d * self.candidate_k + k
                    mask[action_idx] = True
        return mask

    def step(self, action):
        self.steps += 1
        terminated = False
        truncated = False
        step_reward_orig = 0.0
        final_reward = 0.0

        # 解码动作
        dest_idx = action // self.candidate_k
        k = action % self.candidate_k

        # 有效性检查
        if (dest_idx < self.n_destinations and
            k < len(self.candidates[dest_idx]) and
            self.candidates[dest_idx][k] != -1):
            src_idx = self.candidates[dest_idx][k]
            amt_0 = torch.min(self.current_supply[src_idx, 0], self.current_demand[dest_idx, 0]).item()
            amt_1 = torch.min(self.current_supply[src_idx, 1], self.current_demand[dest_idx, 1]).item()
            amount = amt_0 + amt_1
            if amount > 0:
                self.current_supply[src_idx, 0] -= amt_0
                self.current_demand[dest_idx, 0] -= amt_0
                self.current_supply[src_idx, 1] -= amt_1
                self.current_demand[dest_idx, 1] -= amt_1
                min_dist = self.distances[src_idx, dest_idx].item()
                trans_box_mileage = min_dist * amount
                self.total_box_mileage += trans_box_mileage
                self.step_mileage_sum += min_dist
                self.dest_total_mileage[dest_idx] += trans_box_mileage

                self.allocation_history.append((src_idx, dest_idx, amount))
                if len(self.allocation_history) > self.max_history_len:
                    self.allocation_history = self.allocation_history[-self.max_history_len:]

                if self.current_demand[dest_idx].sum() <= 1e-3 and dest_idx not in self.fulfilled_dest_order:
                    self.fulfilled_dest_order.append(dest_idx)
                    self.fulfilled_dest_dist.append(min_dist)
                if self.current_demand[dest_idx].sum() <= 1e-3 and self.completion_step[dest_idx] == -1:
                    self.completion_step[dest_idx] = self.steps

                self.transport_history.append({
                    'source': src_idx,
                    'destination': dest_idx,
                    'amount': amount,
                    'distance_per_box': min_dist,
                    'step_distance': min_dist,
                    'box_distance': trans_box_mileage,
                    'step': self.steps
                })

                # 单步奖励（与原公式一致）
                step_reward_orig = 200 * amount * self.box_base / (min_dist * self.total_boxes + 1e-6)

                # 专注状态更新
                if self.current_focus_dest is None:
                    self.current_focus_dest = dest_idx
                    self.dest_partially_served = True
                if self.current_demand[dest_idx].sum() <= 1e-3:
                    self.current_focus_dest = None
                    self.dest_partially_served = False

                # 更新候选集（因为供应量变化）
                self._update_candidates()

                if torch.sum(self.current_demand) <= 1e-3:
                    terminated = True

        self.cumulative_step_reward += step_reward_orig
        if self.steps >= self.max_steps:
            truncated = True

        # 最终奖励（分母为步骤总里程）
        if terminated or truncated:
            current_demand_np = self.current_demand.cpu().numpy()
            completed_boxes = self.total_boxes - np.sum(current_demand_np)
            ratio = completed_boxes / self.total_boxes if self.total_boxes > 0 else 1.0
            if terminated:
                denominator = self.step_mileage_sum
                if denominator <= 0:
                    denominator = 1e-6
                final_reward = self.final_base / denominator

            adjusted_total = self.cumulative_step_reward * ratio + final_reward
            correction = adjusted_total - self.cumulative_step_reward
            step_reward_final = step_reward_orig + correction
        else:
            step_reward_final = step_reward_orig
            ratio = None
            correction = 0.0

        info = {
            'step_mileage_sum': self.step_mileage_sum,
            'total_box_mileage': self.total_box_mileage,
            'transport_history': self.transport_history,
            'remaining_demand': float(self.current_demand.sum().item()),
            'scene_idx': self.scene_idx,
            'step_transport_reward': step_reward_orig,
            'final_reward_added': final_reward,
            'correction': correction,
            'ratio': ratio,
        }
        return self._get_state(), step_reward_final, terminated, truncated, info


# ===================== 自定义参数化动作分布类 =====================
# ===================== 自定义参数化动作分布类（鸭子类型实现） =====================
class HierarchicalDistribution:
    """
    参数化动作分布：dest（目的站） + slot（候选槽位）两级选择
    不继承任何基类，仅实现必要方法供策略调用。
    """
    def __init__(self, dest_logits, latent_pi, dest_embedding, slot_head, candidate_k):
        self.dest_logits = dest_logits          # [batch, n_dest]
        self.latent_pi = latent_pi              # [batch, latent_dim]
        self.dest_embedding = dest_embedding    # nn.Embedding
        self.slot_head = slot_head              # nn.Module
        self.candidate_k = candidate_k
        self.n_dest = dest_logits.shape[-1]
        self.device = dest_logits.device

        # 用于存储采样时的分布对象
        self.dest_dist = None
        self.slot_dist = None
        self.sampled_dest = None
        self._action = None

        # 存储批量的二维掩码（由策略传入）
        self.masks_2d = None   # [batch, n_dest, K]

    def set_masks(self, masks_2d):
        """接收二维动作掩码，转换为布尔型以便后续使用"""
        if masks_2d is not None:
            if not isinstance(masks_2d, torch.Tensor):
                masks_2d = torch.as_tensor(masks_2d, device=self.device)
            # 确保掩码为布尔类型
            self.masks_2d = masks_2d.bool()
            # 应用目的站掩码（只要该目的站有任何一个槽位可用）
            mask_dest = self.masks_2d.any(dim=-1)  # [batch, n_dest]
            self.dest_logits = self.dest_logits.masked_fill(~mask_dest, -float('inf'))
        else:
            self.masks_2d = None

    def _get_slot_logits(self, dest):
        dest_emb = self.dest_embedding(dest)
        combined = torch.cat([self.latent_pi, dest_emb], dim=-1)
        return self.slot_head(combined)

    def sample(self):
        # 采样目的站
        self.dest_dist = Categorical(logits=self.dest_logits)
        dest = self.dest_dist.sample()
        self.sampled_dest = dest

        # 生成条件槽位 logits
        slot_logits = self._get_slot_logits(dest)
        # 应用槽位掩码
        if self.masks_2d is not None:
            batch_idx = torch.arange(dest.shape[0], device=self.device)
            mask_slot = self.masks_2d[batch_idx, dest]  # [batch, K]
            slot_logits = slot_logits.masked_fill(~mask_slot, -float('inf'))

        self.slot_dist = Categorical(logits=slot_logits)
        slot = self.slot_dist.sample()

        # 组合动作索引（环境需要的格式）
        action = dest * self.candidate_k + slot
        self._action = action
        return action

    def mode(self):
        # 确定性选择：argmax dest, argmax slot
        with torch.no_grad():
            dest = torch.argmax(self.dest_logits, dim=-1)
            slot_logits = self._get_slot_logits(dest)
            if self.masks_2d is not None:
                batch_idx = torch.arange(dest.shape[0], device=self.device)
                mask_slot = self.masks_2d[batch_idx, dest]
                slot_logits = slot_logits.masked_fill(~mask_slot, -float('inf'))
            slot = torch.argmax(slot_logits, dim=-1)
        return dest * self.candidate_k + slot

    def log_prob(self, actions):
        dest = actions // self.candidate_k
        slot = actions % self.candidate_k

        # 目的站对数概率
        dest_log_prob = Categorical(logits=self.dest_logits).log_prob(dest)

        # 槽位对数概率
        slot_logits = self._get_slot_logits(dest)
        if self.masks_2d is not None:
            batch_idx = torch.arange(dest.shape[0], device=self.device)
            mask_slot = self.masks_2d[batch_idx, dest]  # 已经是布尔型
            slot_logits = slot_logits.masked_fill(~mask_slot, -float('inf'))
        slot_log_prob = Categorical(logits=slot_logits).log_prob(slot)

        return dest_log_prob + slot_log_prob

    def entropy(self):
        dest_dist = Categorical(logits=self.dest_logits)
        dest_entropy = dest_dist.entropy()

        all_dests = torch.arange(self.n_dest, device=self.device)
        latent_expanded = self.latent_pi.unsqueeze(1).expand(-1, self.n_dest, -1)
        dest_emb_all = self.dest_embedding(all_dests).unsqueeze(0).expand(latent_expanded.shape[0], -1, -1)
        combined_all = torch.cat([latent_expanded, dest_emb_all], dim=-1)
        slot_logits_all = self.slot_head(combined_all)

        if self.masks_2d is not None:
            # self.masks_2d 已经是布尔型
            slot_logits_all = slot_logits_all.masked_fill(~self.masks_2d, -float('inf'))
            all_invalid = ~self.masks_2d.any(dim=-1)  # 布尔取反，正常工作
            if all_invalid.any():
                slot_logits_all[all_invalid] = 0.0

        slot_dist_all = Categorical(logits=slot_logits_all)
        slot_entropy_given_dest = slot_dist_all.entropy()

        dest_probs = F.softmax(self.dest_logits, dim=-1)
        expected_slot_entropy = (dest_probs * slot_entropy_given_dest).sum(dim=-1)

        return dest_entropy + expected_slot_entropy

    def get_actions(self):
        return self._action

    def actions_from_params(self, dest, slot):
        return dest * self.candidate_k + slot


# ===================== 自定义参数化策略类 =====================
class HierarchicalMaskablePolicy(MaskableActorCriticPolicy):
    """
    继承 MaskableActorCriticPolicy，重写动作分布生成逻辑以支持参数化动作空间。
    额外参数：
        n_destinations: 目的站数量
        candidate_k: 候选槽位数
    """
    def __init__(self, *args, n_destinations, candidate_k, **kwargs):
        self.n_destinations = n_destinations
        self.candidate_k = candidate_k
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        latent_dim = self.mlp_extractor.latent_dim_pi

        # 目的站 logits 头
        self.dest_head = nn.Linear(latent_dim, self.n_destinations)

        # 目的站嵌入层（用于条件槽位选择）
        self.dest_embedding = nn.Embedding(self.n_destinations, 32)

        # 槽位 logits 头（输入 latent + dest_emb）
        self.slot_head = nn.Sequential(
            nn.Linear(latent_dim + 32, 64),
            nn.ReLU(),
            nn.Linear(64, self.candidate_k)
        )

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor, latent_sde=None):
        # 计算目的站 logits
        dest_logits = self.dest_head(latent_pi)   # [batch, n_dest]

        # 创建自定义分布对象
        dist = HierarchicalDistribution(
            dest_logits=dest_logits,
            latent_pi=latent_pi,
            dest_embedding=self.dest_embedding,
            slot_head=self.slot_head,
            candidate_k=self.candidate_k
        )
        return dist

    def _ensure_tensor_and_device(self, action_masks, device):
        if action_masks is None:
            return None
        if not isinstance(action_masks, torch.Tensor):
            action_masks = torch.as_tensor(action_masks, device=device)
        else:
            action_masks = action_masks.to(device)
        return action_masks

    def forward(self, obs: torch.Tensor, deterministic: bool = False, action_masks=None):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf)

        distribution = self._get_action_dist_from_latent(latent_pi)

        # 处理动作掩码：转换为张量并 reshape
        action_masks = self._ensure_tensor_and_device(action_masks, obs.device)
        if action_masks is not None:
            batch_size = action_masks.shape[0]
            masks_2d = action_masks.view(batch_size, self.n_destinations, self.candidate_k)
            distribution.set_masks(masks_2d)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        log_prob = distribution.log_prob(action)
        # 注意：熵不在此返回，训练时会通过 evaluate_actions 计算

        return action, values, log_prob

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor, action_masks=None):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf)

        distribution = self._get_action_dist_from_latent(latent_pi)

        action_masks = self._ensure_tensor_and_device(action_masks, obs.device)
        if action_masks is not None:
            batch_size = action_masks.shape[0]
            masks_2d = action_masks.view(batch_size, self.n_destinations, self.candidate_k)
            distribution.set_masks(masks_2d)

        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        return values, log_prob, entropy

    def _predict(self, obs: torch.Tensor, deterministic: bool = False, action_masks=None):
        features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)

        action_masks = self._ensure_tensor_and_device(action_masks, obs.device)
        if action_masks is not None:
            batch_size = action_masks.shape[0]
            masks_2d = action_masks.view(batch_size, self.n_destinations, self.candidate_k)
            distribution.set_masks(masks_2d)

        if deterministic:
            return distribution.mode()
        return distribution.sample()

# ===================== 训练记录回调 =====================
class EpisodeRecorderCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_raw_rewards = []
        self.episode_step_mileages = []
        self.episode_scenes = []
        self.episode_counts = []
        self.episode_completed_boxes = []
        self.current_episode_train_reward = 0
        self.current_episode_raw_total = 0

    def _on_step(self) -> bool:
        train_reward = self.locals['rewards'][0]
        self.current_episode_train_reward += train_reward
        info = self.locals['infos'][0]
        if 'step_transport_reward' in info:
            self.current_episode_raw_total += info['step_transport_reward']
            if self.locals['dones'][0]:
                self.current_episode_raw_total += info.get('final_reward_added', 0)
        if self.locals['dones'][0]:
            env = self.training_env.envs[0]
            while hasattr(env, 'env') and isinstance(env, Monitor):
                env = env.env
            completed_boxes = env.total_boxes - np.sum(env.current_demand.cpu().numpy())
            self.episode_counts.append(len(self.episode_counts))
            self.episode_rewards.append(self.current_episode_train_reward)
            self.episode_raw_rewards.append(self.current_episode_raw_total)
            self.episode_step_mileages.append(info['step_mileage_sum'])
            self.episode_scenes.append(info['scene_idx'])
            self.episode_completed_boxes.append(completed_boxes)
            print(
                f"Step {self.num_timesteps} | Episode {len(self.episode_counts)} finished | Step mileage sum = {info['step_mileage_sum']:.2f}")
            self.current_episode_train_reward = 0
            self.current_episode_raw_total = 0
        return True


# ===================== 绘图函数 =====================
def plot_training_curves(episode_data, save_path=None):
    counts = np.array(episode_data['counts'])
    rewards = np.array(episode_data['rewards'])
    step_mileages = np.array(episode_data['step_mileages'])
    scenes = np.array(episode_data['scenes'])
    mask0 = scenes == 0
    mask1 = scenes == 1
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    if np.any(mask0):
        plt.scatter(counts[mask0], rewards[mask0], color='blue', alpha=0.7, s=30)
    if np.any(mask1):
        plt.scatter(counts[mask1], rewards[mask1], color='orange', alpha=0.7, s=30)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.subplot(1, 2, 2)
    if np.any(mask0):
        plt.scatter(counts[mask0], step_mileages[mask0], color='blue', alpha=0.7, s=30)
    if np.any(mask1):
        plt.scatter(counts[mask1], step_mileages[mask1], color='orange', alpha=0.7, s=30)
    plt.xlabel('Episode')
    plt.ylabel('Total Step Mileage')
    plt.title('Total Step Mileage per Episode')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ===================== 测试函数 =====================
def evaluate_scenario(model_path, vec_norm_path, station_names, distance_matrix,
                      supply, demand, scene_name, candidate_k=3):
    test_env = ContainerEnv(station_names, distance_matrix,
                            max_steps=500, supply=supply, demand=demand,
                            candidate_k=candidate_k)
    test_env = Monitor(test_env)
    orig_env = test_env
    while hasattr(orig_env, 'env'):
        orig_env = orig_env.env
    test_vec_env = DummyVecEnv([lambda: test_env])
    test_vec_env = VecNormalize.load(vec_norm_path, test_vec_env)
    test_vec_env.training = False
    test_vec_env.norm_reward = False
    loaded_model = MaskablePPO.load(model_path, env=test_vec_env)
    obs = test_vec_env.reset()
    total_reward_norm = 0
    total_step_reward_raw = 0
    total_final_reward = 0
    done = False
    step_count = 0
    final_ratio = 1.0
    transport_history = []
    print(f"\n===== 测试 {scene_name} 场景 =====")
    print("运输详情：")
    while not done:
        masks = test_vec_env.env_method("action_masks")[0]
        action, _states = loaded_model.predict(obs, deterministic=True, action_masks=masks)
        obs, reward_arr, done_arr, infos = test_vec_env.step(action)
        reward_norm = reward_arr[0]
        info = infos[0]
        total_reward_norm += reward_norm
        if 'step_transport_reward' in info:
            total_step_reward_raw += info['step_transport_reward']
        if 'final_reward_added' in info:
            total_final_reward += info['final_reward_added']
        completed_boxes = orig_env.total_boxes - np.sum(orig_env.current_demand.cpu().numpy())
        final_ratio = completed_boxes / orig_env.total_boxes if orig_env.total_boxes > 0 else 1.0
        if len(info['transport_history']) > step_count:
            new = info['transport_history'][step_count:]
            transport_history.extend(new)
            for t in new:
                s_name = station_names[t['source']]
                d_name = station_names[t['destination']]
                print(f"Step {t['step']}: {s_name} -> {d_name} | {t['amount']} 箱 | 步骤距离 {t['step_distance']:.1f}")
            step_count = len(info['transport_history'])
        done = done_arr[0]
    print(f"\n=== {scene_name} 测试结果 ===")
    print(f"归一化总奖励: {total_reward_norm:.2f}")
    print(f"原始步奖励累加: {total_step_reward_raw:.2f}")
    print(f"原始最终奖励: {total_final_reward:.2f}")
    print(f"完成比例: {final_ratio:.2f}")
    print(f"修正项: {info.get('correction', 0):.2f}")
    print(f"调运步骤总里程: {info['step_mileage_sum']:.2f}")
    print(f"剩余需求: {info['remaining_demand']}")
    print(f"运输次数: {len(transport_history)}")


# ===================== 主训练流程 =====================
if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"模型将使用设备: {device}")
    timesteps = 500000
    OUTPUT_DIR = "../强化学习-3天/4-17"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stations, dist_matrix = load_stations_and_distance()
    supply_40, demand_40 = load_supply_demand(stations)
    SCENES = [(supply_40, demand_40)]

    edge_index = np.loadtxt("./基础数据/simplified_edge_index.txt", delimiter="\t", dtype=np.int64, skiprows=1).T
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr_np = np.loadtxt("./基础数据/simplified_edges.txt", delimiter="\t", dtype=np.float32, skiprows=1,
                              usecols=2)
    edge_attr_np = np.repeat(edge_attr_np, 2)
    edge_attr = torch.tensor(edge_attr_np, dtype=torch.float32).unsqueeze(-1)
    n_nodes = len(stations)
    node_feat_dim = 9
    gnn_output_dim = 128

    candidate_k = 3  # Top-K 候选源站数量

    def make_env(scene_list, stations, dist_matrix, k):
        def _init():
            env = ContainerEnv(stations, dist_matrix, max_steps=2 * len(stations),
                               scene_list=scene_list, candidate_k=k)
            env = Monitor(env)
            return env
        return _init

    vec_env = make_vec_env(make_env(SCENES, stations, dist_matrix, candidate_k),
                           n_envs=1, vec_env_cls=DummyVecEnv)
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=100.0,
        gamma=0.998
    )

    # 自定义策略参数字典
    policy_kwargs = dict(
        features_extractor_class=PyG_GNN_Extractor,
        features_extractor_kwargs=dict(
            features_dim=gnn_output_dim,
            n_nodes=n_nodes,
            node_feat_dim=node_feat_dim,
            edge_index=edge_index,
            edge_attr=edge_attr,
            gnn_hidden=128
        ),
        activation_fn=nn.LeakyReLU,
        net_arch=dict(pi=[512, 256, 128], vf=[512, 384, 256]),
        ortho_init=True,
        # 新增参数化动作空间所需参数
        n_destinations=n_nodes,
        candidate_k=candidate_k,
    )

    model = MaskablePPO(
        policy=HierarchicalMaskablePolicy,   # 使用自定义参数化策略
        env=vec_env,
        tensorboard_log="./tensorboard_logs/",
        device=device,
        learning_rate=cosine_annealing_lr_with_restart(),
        n_steps=3072,
        batch_size=1024,
        gamma=0.998,
        gae_lambda=0.98,
        clip_range=lambda progress: 0.3 * (0.7 + 0.3 * progress),
        ent_coef=0.03,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
    )

    ent_callback = NonLinearEntCoefCallback(
        initial_ent_coef=0.06,
        mid_ent_coef=0.03,
        final_ent_coef=0.02,
        total_timesteps=timesteps,
        mid_ratio=0.5,
        final_ratio=0.9
    )
    recorder_callback = EpisodeRecorderCallback(verbose=1)
    callbacks = [ent_callback, recorder_callback]

    try:
        model.learn(total_timesteps=timesteps, callback=callbacks)
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n显存溢出！尝试清理显存后继续...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("如果仍然报错，请减小batch_size或net_arch规模")
            raise e

    model.save(os.path.join(OUTPUT_DIR, "ppo_container_model"))
    vec_env.save(os.path.join(OUTPUT_DIR, "vec_normalize_params.pkl"))

    episode_data = {
        'counts': recorder_callback.episode_counts,
        'rewards': recorder_callback.episode_raw_rewards,
        'step_mileages': recorder_callback.episode_step_mileages,
        'scenes': recorder_callback.episode_scenes
    }
    plot_training_curves(episode_data, save_path=os.path.join(OUTPUT_DIR, "training_curves.png"))

    evaluate_scenario(
        os.path.join(OUTPUT_DIR, "ppo_container_model"),
        os.path.join(OUTPUT_DIR, "vec_normalize_params.pkl"),
        stations, dist_matrix, supply_40, demand_40, "40GP",
        candidate_k=candidate_k
    )