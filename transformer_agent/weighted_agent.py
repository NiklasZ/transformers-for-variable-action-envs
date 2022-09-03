from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F

from transformer_agent.base_agent import layer_init, Agent


class WeightedCritic(nn.Module):
    def __init__(self, device, embed_size, envs):
        super(WeightedCritic, self).__init__()
        self.device = device
        self.embed_size = embed_size
        self.unit_action_space = envs.action_space.nvec[1:].sum()
        self.decoder = layer_init(nn.Linear(self.embed_size, 1), std=1)
        self.value_calc = layer_init(nn.Linear(6, 1), std=1)

    # x: network's output: [seqlength(V), batch_dim, embed_size]
    # entity_mask: [batch_dim, V]
    # (V is the max number of units across all envs).
    # output: [num_envs]
    def forward(self, x, player_unit_mask, opponent_unit_mask, neutral_unit_mask):
        x_reshaped = x.permute((1, 0, 2))  # [V, N, embed_size] -> [N, V, embed_size]
        N, V, embed_size = x_reshaped.shape
        individual_values = torch.squeeze(self.decoder(x_reshaped))
        weightable_values = torch.zeros((N, 6)).to(self.device)

        # Take sums and averages for player, opponent and neutral units
        weightable_values[:, 0] = torch.sum(individual_values.masked_fill(player_unit_mask, 0), axis=1)
        weightable_values[:, 1] = torch.mean(individual_values.masked_fill(player_unit_mask, 0), axis=1)
        weightable_values[:, 2] = torch.sum(individual_values.masked_fill(opponent_unit_mask, 0), axis=1)
        weightable_values[:, 3] = torch.mean(individual_values.masked_fill(opponent_unit_mask, 0), axis=1)
        weightable_values[:, 4] = torch.sum(individual_values.masked_fill(neutral_unit_mask, 0), axis=1)
        weightable_values[:, 5] = torch.mean(individual_values.masked_fill(neutral_unit_mask, 0), axis=1)

        out = self.value_calc(weightable_values)
        return out


class WeightedAgent(Agent):
    def __init__(self, map_size, envs, device, num_layers=5, dim_feedforward=512, num_heads=7, padding=0):
        super(WeightedAgent, self).__init__(map_size, envs, device, num_layers, dim_feedforward, num_heads, padding)
        self.critic = WeightedCritic(device, self.padded_size, envs)

    def get_value(self, x, entity_mask, entity_count, player_unit_mask, enemy_unit_mask, neutral_unit_mask):
        max_units_in_batch = torch.max(entity_count)
        trimmed_x = x[:, :max_units_in_batch, :]
        trimmed_mask = entity_mask[:, :max_units_in_batch]
        trimmed_player = player_unit_mask[:, :max_units_in_batch]
        trimmed_enemy = enemy_unit_mask[:, :max_units_in_batch]
        trimmed_neutral = neutral_unit_mask[:, :max_units_in_batch]

        return self.critic(self.forward(trimmed_x, trimmed_mask), trimmed_player, trimmed_enemy, trimmed_neutral)


def reshape_observation_extended(x: torch.Tensor, device: str) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Converts original Micro-RTS observation into datastructures necessary to the transformer.
    :param x: [batch_dim,height,width,observation_state]
    :param device: cpu or gpu device
    :return:
    reshaped observation [batch_dim, height * width, embed_size]
    boolean padding mask (pass on to transformer) [batch_dim, height * width]
    all entity counts (players and resource) [batch_dim]
    player unit positions bitmap [batch_dim, height * width]
    player unit mask [batch_dim, height * width]
    enemy unit mask [batch_dim, height * width]
    neutral unit mask [batch_dim, height * width]
    """
    # indices in observation where unit and resource positions are encoded.
    player_1, player_2, resource = 11, 12, 14
    N, H, W, C = x.shape
    out = x.view(N, H * W, C)
    x_reshaped = torch.zeros(N, H * W, H * W + C).to(device)
    entity_mask = torch.ones(N, H * W, dtype=torch.bool).to(device)
    player_unit_mask = torch.ones(N, H * W, dtype=torch.bool).to(device)
    enemy_unit_mask = torch.ones(N, H * W, dtype=torch.bool).to(device)
    neutral_unit_mask = torch.ones(N, H * W, dtype=torch.bool).to(device)
    entity_pos = [torch.cat((o[:, player_1].nonzero(),
                             o[:, player_2].nonzero(),
                             o[:, resource].nonzero())) for o in out]
    entity_unit_counts = (out[:, :, player_1].count_nonzero(axis=1) +
                          out[:, :, player_2].count_nonzero(axis=1) +
                          out[:, :, resource].count_nonzero(axis=1)).to(device)
    player_unit_counts = out[:, :, player_1].count_nonzero(axis=1)
    enemy_unit_indices = player_unit_counts + out[:, :, player_2].count_nonzero(axis=1)
    neutral_unit_indices = enemy_unit_indices + out[:, :, resource].count_nonzero(axis=1)

    for i in range(N):
        num_entities = entity_pos[i].shape[0]
        x_reshaped[i, :num_entities, :H * W] = F.one_hot(entity_pos[i][:, 0], H * W)
        x_reshaped[i, :num_entities, H * W:] = out[i, entity_pos[i][:, 0]]
        entity_mask[i, :num_entities] = False
        player_unit_mask[i, :player_unit_counts[i]] = False
        enemy_unit_mask[i, player_unit_counts[i]:enemy_unit_indices[i]] = False
        neutral_unit_mask[i, enemy_unit_indices[i]:neutral_unit_indices[i]] = False

    player_unit_positions = out[:, :, player_1].to(device)
    return x_reshaped, entity_mask, entity_unit_counts, player_unit_positions, \
           player_unit_mask, enemy_unit_mask, neutral_unit_mask