import torch
from gym.wrappers.monitoring import video_recorder
from torch.utils.tensorboard import SummaryWriter
import argparse
from distutils.util import strtobool
import numpy as np
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
from gym_microrts import microrts_ai
from gym.spaces import MultiDiscrete
import time
import random
import os
from stable_baselines3.common.vec_env import VecEnvWrapper, VecVideoRecorder
import matplotlib.pyplot as plt
import glob
from transformer_agent.mixed_embedded_agent import MixedEmbeddedAgent, reshape_observation_mixed_embedded
from transformer_agent.weighted_agent import WeightedAgent, reshape_observation_extended
from jpype.types import JArray, JInt


def make_if_not_exists(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)


parser = argparse.ArgumentParser(description='Agent evaluation')

subparsers = parser.add_subparsers(help='sub-command help', dest='command')
subparsers.required = True

parser_base = subparsers.add_parser('base', help='for the basic transformer agent')
parser_embedded = subparsers.add_parser('embedded', help='for the transformer agent using an embedding')

# Common arguments
for p in [parser_base, parser_embedded]:
    p.add_argument('--exp-name', type=str, default="transformer",
                   help='the name of this experiment')
    p.add_argument('--seed', type=int, default=1,
                   help='seed of the experiment')
    p.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                   help='if toggled, cuda will not be enabled by default')
    p.add_argument('--wandb-project-name', type=str, default="cleanRL",
                   help="the wandb's project name")
    p.add_argument('--wandb-entity', type=str, default=None,
                   help="the entity (team) of wandb's project")
    p.add_argument('--num-steps', type=int, default=256,
                   help='the number of steps per game environment')
    p.add_argument('--num-eval-runs', type=int, default=10,
                   help='the number of bot game environment; 16 bot envs measn 16 games')
    p.add_argument('--agent-model-path', type=str, required=True,
                   help="the path to the agent's model")
    p.add_argument('--max-steps', type=int, default=2000,
                   help="the maximum number of game steps in microrts")
    p.add_argument('--transformer-layers', type=int, default=5,
                   help='the number of layers to use in the transformer encoder')
    p.add_argument('--attention-heads', type=int, default=7,
                   help='determines number of heads to use for multi-headed attention in transformer')
    p.add_argument('--padding', type=int, default=0,
                   help='how much to pad inputs to the transformer encoder by (can use this to balance out '
                        'heads)')
    p.add_argument('--feed-forward-neurons', type=int, default=512,
                   help='the number of feed-forward neurons in a transformer layer')
    p.add_argument('--map-size', type=int, default=8,
                   help='which size of map to use. Currently supported 8, 16')
    p.add_argument('--prod-mode', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                   help='run the script in production mode and use wandb to log outputs')
    p.add_argument('--record-video', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                   help='whether to capture videos of the agent performances (check out `videos` folder)')
    p.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                   help='if toggled, `torch.backends.cudnn.deterministic=False`')

# Specific to embedded variant
parser_embedded.add_argument('--embed-size', type=int, default=64,
                             help='when using embeddings, determines '
                                  'how large an embedding generated from an observation feature should be.')
args = parser.parse_args()
if not args.seed:
    args.seed = int(time.time())


class VecMonitor(VecEnvWrapper):
    def __init__(self, venv):
        VecEnvWrapper.__init__(self, venv)
        self.eprets = None
        self.eplens = None
        self.epcount = 0
        self.tstart = time.time()

    def reset(self):
        obs = self.venv.reset()
        self.eprets = np.zeros(self.num_envs, 'f')
        self.eplens = np.zeros(self.num_envs, 'i')
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        self.eprets += rews
        self.eplens += 1

        newinfos = list(infos[:])
        for i in range(len(dones)):
            if dones[i]:
                info = infos[i].copy()
                ret = self.eprets[i]
                eplen = self.eplens[i]
                epinfo = {'r': ret, 'l': eplen, 't': round(time.time() - self.tstart, 6)}
                info['episode'] = epinfo
                self.epcount += 1
                self.eprets[i] = 0
                self.eplens[i] = 0
                newinfos[i] = info
        return obs, rews, dones, newinfos


class MicroRTSStatsRecorder(VecEnvWrapper):
    def reset(self):
        obs = self.venv.reset()
        self.raw_rewards = [[] for _ in range(self.num_envs)]
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        for i in range(len(dones)):
            self.raw_rewards[i] += [infos[i]["raw_rewards"]]
        newinfos = list(infos[:])
        for i in range(len(dones)):
            if dones[i]:
                info = infos[i].copy()
                raw_rewards = np.array(self.raw_rewards[i]).sum(0)
                raw_names = [str(rf) for rf in self.rfs]
                info['microrts_stats'] = dict(zip(raw_names, raw_rewards))
                self.raw_rewards[i] = []
                newinfos[i] = info
        return obs, rews, dones, newinfos


class ModifiedVideoRecorder(VecVideoRecorder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def start_video_recorder(self, counter='') -> None:
        self.close_video_recorder()

        video_name = f"evaluation-game-{counter}"
        base_path = os.path.join(self.video_folder, video_name)
        self.video_recorder = video_recorder.VideoRecorder(
            env=self.env, base_path=base_path, metadata={"step_id": self.step_id}
        )

        self.video_recorder.capture_frame()
        self.recorded_frames = 1
        self.recording = True


# TRY NOT TO MODIFY: setup the environment
experiment_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
print(f'Running experiment {experiment_name}')

writer = SummaryWriter(f"runs/{experiment_name}")
writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
    '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))

if args.prod_mode:
    import wandb

    run = wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True,
                     config=vars(args), name=experiment_name, save_code=True)
    writer = SummaryWriter(f"/tmp/{experiment_name}")

# TRY NOT TO MODIFY: seeding
device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

all_ais = {
    "guidedRojoA3N": microrts_ai.guidedRojoA3N,
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
}
ai_names, ais = list(all_ais.keys()), list(all_ais.values())
ai_match_stats = dict(zip(ai_names, np.zeros((len(ais), 3))))
args.num_envs = len(ais)
ai_envs = []

if args.map_size == 8:
    map_path = "maps/8x8/basesWorkers8x8.xml"
elif args.map_size == 16:
    map_path = "maps/16x16/basesWorkers16x16.xml"
else:
    raise Exception(f'Unsupported map size {args.map_size}')

for i in range(len(ais)):
    env = MicroRTSGridModeVecEnv(
        num_bot_envs=1,
        num_selfplay_envs=0,
        max_steps=args.max_steps,
        render_theme=2,
        ai2s=[ais[i]],
        map_path=map_path,
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
    )
    env = MicroRTSStatsRecorder(env)
    env = VecMonitor(env)
    if args.record_video:
        # Note, we don't use an episode step-trigger as episodes have a variable number of steps.
        env = ModifiedVideoRecorder(env, f'videos/{experiment_name}/{ai_names[i]}',
                                    record_video_trigger=lambda x: False, video_length=2000)
    ai_envs += [env]
assert isinstance(env.action_space, MultiDiscrete), "only MultiDiscrete action space is supported"

mapsize = args.map_size ** 2
if args.command == 'base':
    agent = WeightedAgent(mapsize, env, device, args.transformer_layers, args.feed_forward_neurons,
                          args.attention_heads, args.padding).to(device)
    feature_map = reshape_observation_extended
elif args.command == 'embedded':
    agent = MixedEmbeddedAgent(mapsize, env, device, args.transformer_layers, args.feed_forward_neurons,
                               args.attention_heads, args.padding, args.embed_size).to(device)
    feature_map = reshape_observation_mixed_embedded
else:
    raise Exception(f'Unknown command {args.command}')

agent.load_state_dict(torch.load(args.agent_model_path, map_location=device))
agent.eval()

print("Model's state_dict:")
for param_tensor in agent.state_dict():
    print(param_tensor, "\t", agent.state_dict()[param_tensor].size())
total_params = sum([param.nelement() for param in agent.parameters()])
print("Model's total parameters:", total_params)
writer.add_scalar(f"charts/total_parameters", total_params, 0)
# ALGO Logic: Storage for epoch data
action_space_shape = (mapsize, env.action_space.shape[0] - 1)
invalid_action_shape = (mapsize, env.action_space.nvec[1:].sum() + 1)

obs = torch.zeros((args.num_steps, args.num_envs) + env.observation_space.shape).to(device)
actions = torch.zeros((args.num_steps, args.num_envs) + action_space_shape).to(device)
logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
values = torch.zeros((args.num_steps, args.num_envs)).to(device)
invalid_action_masks = torch.zeros((args.num_steps, args.num_envs) + invalid_action_shape).to(device)
# TRY NOT TO MODIFY: start the game
global_step = 0
start_time = time.time()
entity_counts = mapsize * [0]

for envs_idx, env in enumerate(ai_envs):
    next_done = torch.zeros(args.num_envs).to(device)


    for g in range(args.num_eval_runs):
        next_obs, next_entity_mask, next_entity_count, next_unit_position, next_unit_mask, next_enemy_unit_mask, next_neutral_unit_mask = \
            feature_map(torch.Tensor(env.reset()).to(device), device)
        entity_counts[next_entity_count.cpu().numpy()[0]] += 1
        done = False
        if args.record_video:
            env.start_video_recorder(g + 1)

        while not done:
            env.render()
            # ALGO LOGIC: put action logic here
            with torch.no_grad():
                action, logproba, _, invalid_action_mask = agent.get_action(next_obs,
                                                                            next_entity_mask,
                                                                            next_entity_count,
                                                                            next_unit_position,
                                                                            next_unit_mask,
                                                                            envs=env)
            # TRY NOT TO MODIFY: execute the game and log data.
            # the real action adds the source units
            real_action = torch.cat([
                torch.stack(
                    [torch.arange(0, mapsize, device=device) for i in range(env.num_envs)
                     ]).unsqueeze(2), action], 2)

            # at this point, the `real_action` has shape (num_envs, map_height*map_width, 8)
            # so as to predict an action for each cell in the map; this obviously include a
            # lot of invalid actions at cells for which no source units exist, so the rest of
            # the code removes these invalid actions to speed things up
            real_action = real_action.cpu().numpy()
            valid_actions = real_action[invalid_action_mask[:, :, 0].bool().cpu().numpy()]
            valid_actions_counts = invalid_action_mask[:, :, 0].sum(1).long().cpu().numpy()
            java_valid_actions = []
            valid_action_idx = 0
            for env_idx, valid_action_count in enumerate(valid_actions_counts):
                java_valid_action = []
                for c in range(valid_action_count):
                    java_valid_action += [JArray(JInt)(valid_actions[valid_action_idx])]
                    valid_action_idx += 1
                java_valid_actions += [JArray(JArray(JInt))(java_valid_action)]
            java_valid_actions = JArray(JArray(JArray(JInt)))(java_valid_actions)

            try:
                raw_obs, rs, ds, infos = env.step(java_valid_actions)
                next_obs, next_entity_mask, next_entity_count, next_unit_position, next_unit_mask, next_enemy_unit_mask, next_neutral_unit_mask = \
                    feature_map(torch.Tensor(raw_obs).to(device), device)
                entity_counts[next_entity_count.cpu().numpy()[0]] += 1
                # raw_obs, rs, ds, infos = env.step(java_valid_actions)
                # next_obs, next_entity_mask, next_entity_count, next_unit_position, next_unit_mask = feature_map(
                #     torch.Tensor(raw_obs).to(device), device)
            except Exception as e:
                e.printStackTrace()
                # The guidedRojoA3N will sometimes crash. There's not much we can do about this as the algorithm is
                # implemented in the micro-rts environment. To deal with this, we stop the episode and call it a draw.
                if ai_names[envs_idx] == 'guidedRojoA3N':
                    ai_match_stats[ai_names[envs_idx]][1] += 1
                    break
                else:
                    raise

            info = infos[0]
            # When an episode finishes
            if 'episode' in info.keys():
                print("against", ai_names[envs_idx], info['microrts_stats']['WinLossRewardFunction'])
                if info['microrts_stats']['WinLossRewardFunction'] == -1.0:
                    ai_match_stats[ai_names[envs_idx]][0] += 1
                elif info['microrts_stats']['WinLossRewardFunction'] == 0.0:
                    ai_match_stats[ai_names[envs_idx]][1] += 1
                elif info['microrts_stats']['WinLossRewardFunction'] == 1.0:
                    ai_match_stats[ai_names[envs_idx]][2] += 1

                done = True
                if args.record_video:
                    env.close_video_recorder()

        for (label, val) in zip(["loss", "tie", "win"], ai_match_stats[ai_names[envs_idx]]):
            writer.add_scalar(f"charts/{ai_names[envs_idx]}/{label}", val, 0)
        if args.prod_mode and args.record_video:
            video_files = glob.glob(f'videos/{experiment_name}/{ai_names[envs_idx]}/*.mp4')
            for video_file in video_files:
                print(video_file)
                wandb.log({f"RL agent against {ai_names[envs_idx]}": wandb.Video(video_file)})

print(ai_match_stats)
n_rows, n_cols = 3, 5
fig = plt.figure(figsize=(5 * 3, 4 * 3))
for i, var_name in enumerate(ai_names):
    stats = ai_match_stats[var_name]
    ax = fig.add_subplot(n_rows, n_cols, i + 1)
    ax.bar(["loss", "tie", "win"], stats)
    ax.set_title(var_name)
    ax.set_ylim([0, sum(stats)])
fig.suptitle(args.agent_model_path)
fig.tight_layout()
make_if_not_exists('evaluation')
plt.savefig(f'evaluation/{experiment_name}')

cumulative_match_results = np.array(list(ai_match_stats.values())).sum(0)
cumulative_match_results_rate = cumulative_match_results / cumulative_match_results.sum()
if args.prod_mode:
    wandb.log({"Match results": wandb.Image(fig)})
    for (label, val) in zip(["loss", "tie", "win"], cumulative_match_results):
        writer.add_scalar(f"charts/cumulative_match_results/{label}", val, 0)
    for (label, val) in zip(["loss rate", "tie rate", "win rate"], cumulative_match_results_rate):
        writer.add_scalar(f"charts/cumulative_match_results/{label}", val, 0)
env.close()
writer.close()
