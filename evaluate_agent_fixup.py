# Had to manually record these because the guidedRojoA3N crashed just before the end and re-running evaluate_agent.py takes 3 hours.
import numpy as np
from gym_microrts import microrts_ai
from matplotlib import pyplot as plt

import wandb

all_ais = {
    "randomBiasedAI": microrts_ai.randomBiasedAI,
    "randomAI": microrts_ai.randomAI,
    "passiveAI": microrts_ai.passiveAI,
    "workerRushAI": microrts_ai.workerRushAI,
    "lightRushAI": microrts_ai.lightRushAI,
    "coacAI": microrts_ai.coacAI,
    "naiveMCTSAI": microrts_ai.naiveMCTSAI,
    "mixedBot": microrts_ai.mixedBot,
    "rojo": microrts_ai.rojo,
    "izanagi": microrts_ai.izanagi,
    "tiamat": microrts_ai.tiamat,
    "droplet": microrts_ai.droplet,
    "guidedRojoA3N": microrts_ai.guidedRojoA3N
}
ai_names, ais = list(all_ais.keys()), list(all_ais.values())
# run = wandb.init(project='cleanRL', entity=None, sync_tensorboard=True, name='final-bot-evaluation', save_code=True)
# 8x8 match results
# ai_match_stats = {
#     'coacAI': [6, 0, 94],
#     'droplet': [15, 0, 85],
#     'guidedRojoA3N': [21, 3, 76],
#     'izanagi': [24, 5, 71],
#     'lightRushAI': [0, 0, 100],
#     'mixedBot': [8, 4, 88],
#     'naiveMCTSAI': [21, 16, 63],
#     'passiveAI': [0, 0, 100],
#     'randomAI': [0, 0, 100],
#     'randomBiasedAI': [0, 0, 100],
#     'rojo': [0, 0, 100],
#     'tiamat': [0, 0, 100],
#     'workerRushAI': [1, 0, 99],
# }
# # 8x8 entity distribution
# buckets = [0, 0, 0, 1731, 4080, 6893, 74152, 44127, 118471, 120341, 150400, 59445, 26284, 6607, 3689, 2471, 1769, 1448,
#            393, 91, 254, 257, 114, 73, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# map_size = 8 * 8

# 16x16 match results:
ai_match_stats = {
    'coacAI': [8., 4., 88.],
    'droplet': [53., 19., 28.],
    'guidedRojoA3N': [0., 11., 89.],
    'izanagi': [21., 24., 55.],
    'lightRushAI': [29., 4., 67.],
    'mixedBot': [0., 5., 95.],
    'naiveMCTSAI': [0., 83., 17.],
    'passiveAI': [0, 0, 100],
    'randomAI': [0, 0, 100],
    'randomBiasedAI': [0., 0., 100.],
    'rojo': [1., 0., 99.],
    'tiamat': [3., 0., 97.],
    'workerRushAI': [32., 19., 49.],
}

# 16x16 entity distribution
buckets = [0, 0, 0, 151, 717, 1594, 2736, 3457, 62438, 16230, 64197, 37950, 66288, 55041, 86174, 82154, 121785, 86159,
           94663, 61260, 63908, 47105, 48064, 53798, 48671, 45608, 44109, 35112, 36025, 33051, 30177, 29226, 26196,
           19050, 23258, 18004, 13942, 11427, 3866, 3832, 69, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
map_size = 16*16

cumulative_rates = [0, 0, 0]
print(ai_match_stats)
n_rows, n_cols = 3, 5
fig = plt.figure(figsize=(5 * 3, 4 * 3))
for i, var_name in enumerate(ai_names):
    stats = ai_match_stats[var_name]
    ax = fig.add_subplot(n_rows, n_cols, i + 1)
    ax.bar(["loss", "tie", "win"], stats)
    ax.set_title(var_name)
    ax.set_ylim([0, sum(stats)])
    cumulative_rates[0] += stats[0]
    cumulative_rates[1] += stats[1]
    cumulative_rates[2] += stats[2]

total = sum(cumulative_rates)
print(
    f'Loss rate: {cumulative_rates[0] / total}, Tie rate: {cumulative_rates[1] / total} , Win rate: {cumulative_rates[2] / total}')
fig.suptitle('Transformer AI Win-rates')
fig.tight_layout()

total = sum(buckets)
fig2, ax = plt.subplots()
ax.bar(range(0, map_size), np.array(buckets) / total)  # density=False would make counts
ax.set_xlabel('Number of entities in state')
ax.set_ylabel('Normalised Frequency')
ax.set_title(f'{int(map_size**0.5)}x{int(map_size**0.5)} Map Entity Count Distribution')
fig2.show()

pass
# for (label, val) in zip(["loss", "tie", "win"], cumulative_match_results):
#     writer.add_scalar(f"charts/cumulative_match_results/{label}", val, 0)
# for (label, val) in zip(["loss rate", "tie rate", "win rate"], cumulative_match_results_rate):
#     writer.add_scalar(f"charts/cumulative_match_results/{label}", val, 0)
# wandb.log({"Match results": wandb.Image(fig)})
