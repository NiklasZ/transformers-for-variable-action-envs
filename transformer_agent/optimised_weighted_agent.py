import numpy as np
import torch

from transformer_agent.agent import CategoricalMasked
from transformer_agent.weighted_agent import WeightedAgent


class OptimisedWeightedAgent(WeightedAgent):
    def get_action_and_value(self, x,
                             entity_mask,
                             entity_count,
                             player_unit_position,
                             player_unit_mask,
                             enemy_unit_mask,
                             neutral_unit_mask,
                             action=None, invalid_action_masks=None, envs=None):
        # x_reshaped, bool_mask, player_unit_pos, player_unit_counts = self.reshape_for_transformer(x)
        # There's no point in passing in tensors we will mask out anyway, so we trim them here.
        max_units_in_batch = torch.max(entity_count)
        trimmed_x = x[:, :max_units_in_batch, :]
        trimmed_ent_mask = entity_mask[:, :max_units_in_batch]
        trimmed_unit_mask = player_unit_mask[:, :max_units_in_batch]
        trimmed_enemy_mask = enemy_unit_mask[:, :max_units_in_batch]
        trimmed_neutral_mask = neutral_unit_mask[:, :max_units_in_batch]

        base_out = self.forward(trimmed_x, trimmed_ent_mask)
        value = torch.squeeze(self.critic(base_out, trimmed_unit_mask, trimmed_enemy_mask, trimmed_neutral_mask))
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
        return action, logprob.sum(1).sum(1), entropy.sum(1).sum(1), invalid_action_masks, value