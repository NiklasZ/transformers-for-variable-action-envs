import torch
import torch.nn.functional as F


# Input: [batch_dim,height,width,observation_state]
# Output:
# reshaped observation [batch_dim, height * width, embed_size]
# boolean padding mask (pass on to transformer) [batch_dim, height * width]
# player unit counts [batch_dim]
# player unit positions bitmap [batch_dim, height * width]
def reshape_observation(x, device):
    # indices in observation where unit and resource positions are encoded.
    player_1, player_2, resource = 11, 12, 14
    N, H, W, C = x.shape
    out = x.view(N, H * W, C)
    x_reshaped = torch.zeros(N, H * W, H * W + C).to(device)
    bool_mask = torch.ones(N, H * W, dtype=torch.bool).to(device)
    for i, env_obs in enumerate(out):
        entity_pos = torch.cat((out[i, :, player_1].nonzero(),
                                out[i, :, player_2].nonzero(),
                                out[i, :, resource].nonzero()))
        num_entities = entity_pos.shape[0]
        x_reshaped[i, :num_entities, :H * W] = F.one_hot(entity_pos[:, 0], H * W)
        x_reshaped[i, :num_entities, H * W:] = out[i, entity_pos[:, 0]]
        bool_mask[i, :num_entities] = False
    player_unit_counts = out[:, :, player_1].count_nonzero(axis=1).to(device)
    player_unit_positions = out[:, :, player_1].to(device)
    return x_reshaped, bool_mask,  player_unit_counts, player_unit_positions
