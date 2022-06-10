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
run = wandb.init(project='cleanRL', entity=None, sync_tensorboard=True, name='final-bot-evaluation', save_code=True)
ai_match_stats = {
    'coacAI': [1, 0,  99],
    'droplet': [20, 2,  78],
    'guidedRojoA3N': [10, 3,  80],
    'izanagi': [31, 4,  65],
    'lightRushAI': [0, 0,  100],
    'mixedBot': [7, 2,  91],
    'naiveMCTSAI': [20, 14,  66],
    'passiveAI': [0, 0,  100],
    'randomAI': [0, 0,  100],
    'randomBiasedAI': [0, 0,  100],
    'rojo': [0, 0,  100],
    'tiamat': [0, 0,  100],
    'workerRushAI': [1, 1,  98],
}
print(ai_match_stats)
n_rows, n_cols = 3, 5
fig = plt.figure(figsize=(5 * 3, 4 * 3))
for i, var_name in enumerate(ai_names):
    ax = fig.add_subplot(n_rows, n_cols, i + 1)
    ax.bar(["loss", "tie", "win"], ai_match_stats[var_name])
    ax.set_title(var_name)
fig.suptitle('Transformer AI Win-rates')
fig.tight_layout()
cumulative_match_results = np.array(list(ai_match_stats.values())).sum(0)
cumulative_match_results_rate = cumulative_match_results / cumulative_match_results.sum()
print(cumulative_match_results_rate)
    # for (label, val) in zip(["loss", "tie", "win"], cumulative_match_results):
    #     writer.add_scalar(f"charts/cumulative_match_results/{label}", val, 0)
    # for (label, val) in zip(["loss rate", "tie rate", "win rate"], cumulative_match_results_rate):
    #     writer.add_scalar(f"charts/cumulative_match_results/{label}", val, 0)
wandb.log({"Match results": wandb.Image(fig)})

