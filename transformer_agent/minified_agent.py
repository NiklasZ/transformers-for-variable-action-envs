from typing import Tuple

import numpy as np
import torch
from torch import nn

from transformer_agent.agent import Actor, CategoricalMasked
from transformer_agent.weighted_agent import WeightedAgent, WeightedCritic
from torch.nn import functional as F


# class MinifiedWeightedAgent(WeightedAgent):
#     def __init__(self, map_height, map_width, envs, device, num_layers=5, dim_feedforward=512, num_heads=5, padding=2):
#         super(MinifiedWeightedAgent, self).__init__(map_height * map_width, envs, device, num_layers, dim_feedforward,
#                                                     num_heads, padding)
#         self.input_size = (map_height + map_width + 5 + 5 + 3 + 8 + 6) # 43 on an 8x8 map. 43 is a prime number though.
#         self.padded_size = self.padding + self.input_size # So we add 2 which gives us 45.
#         encoder_layer = nn.TransformerEncoderLayer(d_model=self.padded_size,
#                                                    nhead=self.num_heads,
#                                                    dim_feedforward=dim_feedforward)
#         self.network = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
#         self.actor = Actor(self.padded_size, self.map_size, envs, device)
#         self.critic = WeightedCritic(device, self.padded_size, envs)

class MinifiedWeightedAgent(nn.Module):
    def __init__(self, map_height, map_width, envs, device, num_layers=5, dim_feedforward=512, num_heads=5, padding=2):
        super(MinifiedWeightedAgent, self).__init__()
        self.device = device
        self.map_size = map_height * map_width  # E.g 8*8
        # For our case 8,8,27
        # The first part of the input contains the location of the unit on the map
        # The rest are one-hot encodings of the observation features:
        # hit points, resources, owner, unit types, current action
        # On an 8 x 8 map this is 91
        self.input_size = (map_height + map_width + 5 + 5 + 3 + 8 + 6)  # 43 on an 8x8 map. 43 is a prime number though.
        # How much to pad an input with so the size works for the number of attention heads.
        self.padding = padding
        # Number of neurons in the feedforward layer in a transformer block.
        self.dim_feedforward = dim_feedforward
        # Number of encoding layers
        self.num_layers = num_layers
        # Needs to be picked so the input_size is divisible by it.
        self.num_heads = num_heads
        if (self.input_size + self.padding) % num_heads != 0:
            raise Exception(
                f'The input size of {self.input_size} + padding {self.padding} are not divisible by {self.num_heads}')
        self.padded_size = self.padding + self.input_size # So we add 2 which gives us 45.
        # Ignore dropout for now
        self.dropout = 0
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.padded_size,
                                                   nhead=self.num_heads,
                                                   dim_feedforward=dim_feedforward)
        self.network = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.actor = Actor(self.padded_size, self.map_size, envs, device)
        self.critic = WeightedCritic(self.device, self.padded_size, envs)

    # TODO Type this stuff
    def forward(self, x_reshaped, bool_mask):
        # For some alien reason PyTorch breaks all convention and wants the batch dimension 2nd.
        # [batch_dim, seq_length(V), obs_state] -> # [seq_length(V), batch_dim, obs_state]
        x_padded = F.pad(x_reshaped, (0, self.padding))
        return self.network(x_padded.permute(1, 0, 2), src_key_padding_mask=bool_mask)

    def get_action(self, x,
                   entity_mask,
                   entity_count,
                   player_unit_position,
                   player_unit_mask,
                   action=None, invalid_action_masks=None, envs=None):
        # x_reshaped, bool_mask, player_unit_pos, player_unit_counts = self.reshape_for_transformer(x)
        # There's no point in passing in tensors we will mask out anyway, so we trim them here.
        max_units_in_batch = torch.max(entity_count)
        trimmed_x = x[:, :max_units_in_batch, :]
        trimmed_ent_mask = entity_mask[:, :max_units_in_batch]
        trimmed_unit_mask = player_unit_mask[:, :max_units_in_batch]
        base_out = self.forward(trimmed_x, trimmed_ent_mask)
        logits = self.actor(base_out, player_unit_position, trimmed_unit_mask)
        # OLD CODE - DO NOT CHANGE
        grid_logits = logits.view(-1, envs.action_space.nvec[1:].sum())
        split_logits = torch.split(grid_logits, envs.action_space.nvec[1:].tolist(), dim=1)

        if action is None:
            invalid_action_masks = torch.tensor(np.array(envs.vec_client.getMasks(0))).to(self.device)
            invalid_action_masks = invalid_action_masks.view(-1, invalid_action_masks.shape[-1])
            split_invalid_action_masks = torch.split(invalid_action_masks[:, 1:], envs.action_space.nvec[1:].tolist(),
                                                     dim=1)
            multi_categoricals = [CategoricalMasked(self.device, logits=logits, masks=iam) for (logits, iam) in
                                  zip(split_logits, split_invalid_action_masks)]
            action = torch.stack([categorical.sample() for categorical in multi_categoricals])
        else:
            invalid_action_masks = invalid_action_masks.view(-1, invalid_action_masks.shape[-1])
            action = action.view(-1, action.shape[-1]).T
            split_invalid_action_masks = torch.split(invalid_action_masks[:, 1:], envs.action_space.nvec[1:].tolist(),
                                                     dim=1)
            multi_categoricals = [CategoricalMasked(self.device, logits=logits, masks=iam) for (logits, iam) in
                                  zip(split_logits, split_invalid_action_masks)]
        logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        num_predicted_parameters = len(envs.action_space.nvec) - 1
        logprob = logprob.T.view(-1, self.map_size, num_predicted_parameters)
        entropy = entropy.T.view(-1, self.map_size, num_predicted_parameters)
        action = action.T.view(-1, self.map_size, num_predicted_parameters)
        invalid_action_masks = invalid_action_masks.view(-1, self.map_size, envs.action_space.nvec[1:].sum() + 1)
        return action, logprob.sum(1).sum(1), entropy.sum(1).sum(1), invalid_action_masks

    def get_value(self, x, entity_mask, entity_count, player_unit_mask, enemy_unit_mask, neutral_unit_mask):
        max_units_in_batch = torch.max(entity_count)
        trimmed_x = x[:, :max_units_in_batch, :]
        trimmed_mask = entity_mask[:, :max_units_in_batch]
        trimmed_player = player_unit_mask[:, :max_units_in_batch]
        trimmed_enemy = enemy_unit_mask[:, :max_units_in_batch]
        trimmed_neutral = neutral_unit_mask[:, :max_units_in_batch]

        return self.critic(self.forward(trimmed_x, trimmed_mask), trimmed_player, trimmed_enemy, trimmed_neutral)

    def network_size(self):
        print(f'Main NN params: {sum([p.numel() for p in self.network.parameters()])}')
        print(f'Trainable params: {sum([p.numel() for p in self.network.parameters() if p.requires_grad])}')


def reshape_observation_minified(x: torch.Tensor, device: str) -> Tuple[
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
    x_reshaped = torch.zeros(N, H * W, H + W + C).to(device)
    entity_mask = torch.ones(N, H * W, dtype=torch.bool).to(device)
    player_unit_mask = torch.ones(N, H * W, dtype=torch.bool).to(device)
    enemy_unit_mask = torch.ones(N, H * W, dtype=torch.bool).to(device)
    neutral_unit_mask = torch.ones(N, H * W, dtype=torch.bool).to(device)
    entity_pos = [torch.cat((i[:, :, player_1].nonzero(),
                             i[:, :, player_2].nonzero(),
                             i[:, :, resource].nonzero())) for i in x]
    entity_unit_counts = (out[:, :, player_1].count_nonzero(axis=1) +
                          out[:, :, player_2].count_nonzero(axis=1) +
                          out[:, :, resource].count_nonzero(axis=1)).to(device)
    player_unit_counts = out[:, :, player_1].count_nonzero(axis=1)
    enemy_unit_indices = player_unit_counts + out[:, :, player_2].count_nonzero(axis=1)
    neutral_unit_indices = enemy_unit_indices + out[:, :, resource].count_nonzero(axis=1)

    for i in range(N):
        num_entities = entity_pos[i].shape[0]
        # Assign one-hot version of (h,w) positions to output. Should 2H or 2W in length, where we assume H==W for now.
        x_reshaped[i, :num_entities, :H + W] = F.one_hot(entity_pos[i], H).flatten(start_dim=-2)
        # Assign features of each unit, selecting by position.
        x_reshaped[i, :num_entities, H + W:] = x[0, entity_pos[i][:, 0], entity_pos[i][:, 1]]
        entity_mask[i, :num_entities] = False
        player_unit_mask[i, :player_unit_counts[i]] = False
        enemy_unit_mask[i, player_unit_counts[i]:enemy_unit_indices[i]] = False
        neutral_unit_mask[i, enemy_unit_indices[i]:neutral_unit_indices[i]] = False

    player_unit_positions = out[:, :, player_1].to(device)
    return x_reshaped, entity_mask, entity_unit_counts, player_unit_positions, \
           player_unit_mask, enemy_unit_mask, neutral_unit_mask