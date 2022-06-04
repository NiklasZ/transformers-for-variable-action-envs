import torch
import torch.nn.functional as F


# Input: [batch_dim,height,width,observation_state]
# Output:
# reshaped observation [batch_dim, height * width, embed_size]
# boolean padding mask (pass on to transformer) [batch_dim, height * width]
# player entity counts [batch_dim]
# player unit positions bitmap [batch_dim, height * width]
# player unit mask [batch_dim, height * width]
def reshape_observation(x, device):
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
        # entity_pos = torch.cat((out[i, :, player_1].nonzero(),
        #                         out[i, :, player_2].nonzero(),
        #                         out[i, :, resource].nonzero()))
        num_entities = entity_pos[i].shape[0]
        x_reshaped[i, :num_entities, :H * W] = F.one_hot(entity_pos[i][:, 0], H * W)
        x_reshaped[i, :num_entities, H * W:] = out[i, entity_pos[i][:, 0]]
        entity_mask[i, :num_entities] = False
        player_unit_mask[i, :player_unit_counts[i]] = False

    player_unit_positions = out[:, :, player_1].to(device)
    return x_reshaped, entity_mask, entity_unit_counts, player_unit_positions, player_unit_mask
