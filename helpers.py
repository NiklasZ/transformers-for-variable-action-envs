import re
from typing import Tuple
import torch
import torch.nn.functional as F


def reshape_observation(x: torch.Tensor, device: str) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    """
    # indices in observation where unit and resource positions are encoded.
    player_1, player_2, resource = 11, 12, 14
    N, H, W, C = x.shape
    out = x.view(N, H * W, C)
    x_reshaped = torch.zeros(N, H * W, H * W + C).to(device)
    entity_mask = torch.ones(N, H * W, dtype=torch.bool).to(device)
    player_unit_mask = torch.ones(N, H * W, dtype=torch.bool).to(device)
    entity_pos = [torch.cat((o[:, player_1].nonzero(),
                             o[:, player_2].nonzero(),
                             o[:, resource].nonzero())) for o in out]
    entity_unit_counts = (out[:, :, player_1].count_nonzero(axis=1) +
                          out[:, :, player_2].count_nonzero(axis=1) +
                          out[:, :, resource].count_nonzero(axis=1)).to(device)
    player_unit_counts = out[:, :, player_1].count_nonzero(axis=1)

    for i in range(N):
        num_entities = entity_pos[i].shape[0]
        x_reshaped[i, :num_entities, :H * W] = F.one_hot(entity_pos[i][:, 0], H * W)
        x_reshaped[i, :num_entities, H * W:] = out[i, entity_pos[i][:, 0]]
        entity_mask[i, :num_entities] = False
        player_unit_mask[i, :player_unit_counts[i]] = False

    player_unit_positions = out[:, :, player_1].to(device)
    return x_reshaped, entity_mask, entity_unit_counts, player_unit_positions, player_unit_mask



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


class MessageFilter(object):
    def __init__(self, strings_to_filter, stream):
        self.stream = stream
        self.strings_to_filter = strings_to_filter

    def __getattr__(self, attr_name):
        return getattr(self.stream, attr_name)

    def write(self, data):
        if data not in self.strings_to_filter:
            self.stream.write(data)
            self.stream.flush()

    def flush(self):
        self.stream.flush()
