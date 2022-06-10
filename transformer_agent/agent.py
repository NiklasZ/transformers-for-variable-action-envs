import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F


class CategoricalMasked(Categorical):
    def __init__(self, device, probs=None, logits=None, validate_args=None, masks=[], sw=None):
        self.device = device
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.bool()
            logits = torch.where(self.masks, logits, torch.tensor(-1e+8, device=device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.).to(self.device))
        return -p_log_p.sum(-1)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Critic(nn.Module):
    def __init__(self, embed_size, map_size, envs):
        super(Critic, self).__init__()
        self.embed_size = embed_size
        self.map_size = map_size
        self.unit_action_space = envs.action_space.nvec[1:].sum()
        self.decoder = layer_init(nn.Linear(self.embed_size, 1), std=1)

    # x: network's output: [seqlength(V), batch_dim, embed_size]
    # entity_mask: [batch_dim, V]
    # (V is the max number of units across all envs).
    # output: [num_envs]
    def forward(self, x, entity_mask):
        x_reshaped = x.permute((1, 0, 2))  # [V, N, embed_size] -> [N, V, embed_size]
        value_preds = torch.squeeze(self.decoder(x_reshaped))
        out = torch.sum(value_preds.masked_fill(entity_mask, 0), axis=1)
        return out


class WeightedCritic(nn.Module):
    def __init__(self, device, embed_size, map_size, envs):
        super(WeightedCritic, self).__init__()
        self.device = device
        self.embed_size = embed_size
        self.map_size = map_size
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


class Actor(nn.Module):
    def __init__(self, embed_size, map_size, envs, device):
        super(Actor, self).__init__()
        self.device = device
        self.embed_size = embed_size
        self.map_size = map_size
        self.unit_action_space = envs.action_space.nvec[1:].sum()
        self.decoder = layer_init(nn.Linear(embed_size, self.unit_action_space), std=0.01)

    # x: network's output: [V, num_envs, embed_size]
    # (V is the max number of units across all envs).
    # player_unit_positions: [batch_dim, height * width]
    # player_unit_mask: [batch_dim, V]
    # output: [num_envs, env_height * env_width * unit_action_params]
    def forward(self, x, player_unit_positions, player_unit_mask):
        x_reshaped = x.permute((1, 0, 2))  # [V, N, embed_size] -> [N, V, embed_size]
        N = x_reshaped.shape[0]
        out = torch.zeros(N, self.map_size, self.unit_action_space).to(self.device)
        max_player_units = torch.max((player_unit_mask == False).count_nonzero(axis=1))
        trimmed_unit_mask = player_unit_mask[:, :max_player_units]
        unit_action_x, unit_action_y = (trimmed_unit_mask == False).nonzero(as_tuple=True)
        action_preds = self.decoder(x_reshaped[:, :max_player_units, :])
        unit_pos_x, unit_pos_y = player_unit_positions.nonzero(as_tuple=True)
        # We use masks to find the coordinates of units in the output grid
        # And then assign the corresponding action
        out[unit_pos_x, unit_pos_y, :] = action_preds[unit_action_x, unit_action_y, :]

        y = out.view(N, self.map_size * self.unit_action_space)
        return y


class Agent(nn.Module):
    def __init__(self, map_size, envs, device, num_layers=5, dim_feedforward=512, num_heads=7, padding=0):
        super(Agent, self).__init__()
        self.device = device
        self.map_size = map_size  # E.g 8*8
        # For our case 8,8,27
        # The first part of the input contains the location of the unit on the map
        # The rest are one-hot encodings of the observation features:
        # hit points, resources, owner, unit types, current action
        # On an 8 x 8 map this is 91
        # TODO by fixing this embed size we are limiting model parameter size.
        # TODO I should consider a compacter format that generates an embedding.
        self.input_size = (map_size + 5 + 5 + 3 + 8 + 6)
        # How much to pad an input with so the size works for the number of attention heads.
        self.padding = padding
        # Number of neurons in the feedforward layer in a transformer block.
        self.dim_feedforward = dim_feedforward
        # Number of encoding layers
        self.num_layers = num_layers
        # Needs to be picked so the input_size is divisible by it. E.g 91/7 = 13
        self.num_heads = num_heads
        if (self.input_size + self.padding) % num_heads != 0:
            raise Exception(
                f'The input size of {self.input_size} + padding {self.padding} are not divisible by {self.num_heads}')
        self.padded_size = self.input_size + self.padding
        # Ignore dropout for now
        self.dropout = 0
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.padded_size,
                                                   nhead=self.num_heads,
                                                   dim_feedforward=dim_feedforward)
        self.network = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.actor = Actor(self.padded_size, map_size, envs, device)

        self.critic = Critic(self.padded_size, map_size, envs)

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

    def get_value(self, x, entity_mask, entity_count):
        # x_reshaped, bool_mask, _, player_unit_counts = self.reshape_for_transformer(x)
        max_units_in_batch = torch.max(entity_count)
        trimmed_x = x[:, :max_units_in_batch, :]
        trimmed_mask = entity_mask[:, :max_units_in_batch]
        return self.critic(self.forward(trimmed_x, trimmed_mask), trimmed_mask)

    def network_size(self):
        print(f'Main NN params: {sum([p.numel() for p in self.network.parameters()])}')
        print(f'Trainable params: {sum([p.numel() for p in self.network.parameters() if p.requires_grad])}')


class WeightedAgent(Agent):
    def __init__(self, map_size, envs, device, num_layers=5, dim_feedforward=512, num_heads=7, padding=0):
        super(WeightedAgent, self).__init__(map_size, envs, device, num_layers, dim_feedforward, num_heads, padding)
        self.critic = WeightedCritic(device, self.padded_size, map_size, envs)

    def get_value(self, x, entity_mask, entity_count, player_unit_mask, enemy_unit_mask, neutral_unit_mask):
        max_units_in_batch = torch.max(entity_count)
        trimmed_x = x[:, :max_units_in_batch, :]
        trimmed_mask = entity_mask[:, :max_units_in_batch]
        trimmed_player = player_unit_mask[:, :max_units_in_batch]
        trimmed_enemy = enemy_unit_mask[:, :max_units_in_batch]
        trimmed_neutral = neutral_unit_mask[:, :max_units_in_batch]

        return self.critic(self.forward(trimmed_x, trimmed_mask), trimmed_player, trimmed_enemy, trimmed_neutral)


class EmbeddedAgent(WeightedAgent):
    def __init__(self, map_size, envs, device, num_layers=5, dim_feedforward=512, embed_factor=5, num_heads=7, padding=0):
        super(EmbeddedAgent, self).__init__(map_size, envs, device, num_layers, dim_feedforward, num_heads, padding)
        self.embed_factor = embed_factor

        self.embedder = nn.Embedding(2, embed_factor)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.padded_size * embed_factor,
                                                   nhead=self.num_heads,
                                                   dim_feedforward=dim_feedforward)
        self.network = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.actor = Actor(self.padded_size * embed_factor, map_size, envs, device)
        self.critic = WeightedCritic(device, self.padded_size * embed_factor, map_size, envs)

    def forward(self, x_reshaped, bool_mask):
        N, S, _ = x_reshaped.shape
        embedded = self.embedder(x_reshaped.type(torch.IntTensor).to(self.device))
        # [batch_dim, seq_length(V), obs_state, embedding] -> [seq_length(V), batch_dim, obs_state * embedding]
        rearranged = embedded.reshape(N, S, -1).permute(1, 0, 2)
        return self.network(rearranged, src_key_padding_mask=bool_mask)
