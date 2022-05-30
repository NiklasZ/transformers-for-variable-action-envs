# http://proceedings.mlr.press/v97/han19a/han19a.pdf

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import argparse
from distutils.util import strtobool
import numpy as np
import gym
import gym_microrts
from gym.wrappers import TimeLimit, Monitor
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
from gym_microrts import microrts_ai
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os
from stable_baselines3.common.vec_env import VecEnvWrapper, VecVideoRecorder

from helpers import reshape_observation
from transformer import Encoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PPO agent')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="MicrortsDefeatCoacAIShaped-v3",
                        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=100000000,
                        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--prod-mode', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--capture-video', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
                        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument('--resume', type=str, default=None,
                        help='if used, supply the ID of the run to resume.')

    # Algorithm specific arguments
    parser.add_argument('--n-minibatch', type=int, default=4,
                        help='the number of mini batch')
    parser.add_argument('--num-bot-envs', type=int, default=24,
                        help='the number of bot game environment; 16 bot envs means 16 games')
    parser.add_argument('--num-selfplay-envs', type=int, default=0,
                        help='the number of self play envs; 16 self play envs means 8 games')
    parser.add_argument('--num-steps', type=int, default=256,
                        help='the number of steps per game environment')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='the discount factor gamma')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='the lambda for the general advantage estimation')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                        help="coefficient of the entropy")
    parser.add_argument('--vf-coef', type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='the maximum norm for the gradient clipping')
    parser.add_argument('--clip-coef', type=float, default=0.1,
                        help="the surrogate clipping coefficient")
    parser.add_argument('--update-epochs', type=int, default=4,
                        help="the K epochs to update the policy")
    parser.add_argument('--kle-stop', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='If toggled, the policy updates will be early stopped w.r.t target-kl')
    parser.add_argument('--kle-rollback', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='If toggled, the policy updates will roll back to previous policy if KL exceeds target-kl')
    parser.add_argument('--target-kl', type=float, default=0.03,
                        help='the target-kl variable that is referred by --kl')
    parser.add_argument('--gae', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='Use GAE for advantage computation')
    parser.add_argument('--norm-adv', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help="Toggles advantages normalization")
    parser.add_argument('--anneal-lr', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument('--clip-vloss', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='Toggles wheter or not to use a clipped loss for the value function, as per the paper.')

    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())
args.num_envs = args.num_selfplay_envs + args.num_bot_envs
args.batch_size = int(args.num_envs * args.num_steps)
args.minibatch_size = int(args.batch_size // args.n_minibatch)


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
    def __init__(self, env, gamma):
        super().__init__(env)
        self.gamma = gamma

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


# TRY NOT TO MODIFY: setup the environment
experiment_name = f"{args.gym_id}_{args.exp_name}_{args.seed}_{int(time.time())}"
writer = SummaryWriter(f"runs/{experiment_name}")
writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
    '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))
if args.prod_mode:
    import wandb

    run_id = args.resume or wandb.util.generate_id()
    resume = 'must' if args.resume else False
    experiment_name += f'_{run_id}'
    print(f"Running experiment '{experiment_name}'")
    run = wandb.init(id=run_id, resume=resume,
                     project=args.wandb_project_name, entity=args.wandb_entity,
                     # sync_tensorboard=True,
                     config=vars(args), name=experiment_name, monitor_gym=True, save_code=True)
    wandb.tensorboard.patch(save=False)
    writer = SummaryWriter(f"/tmp/{experiment_name}")
    CHECKPOINT_FREQUENCY = 50

# TRY NOT TO MODIFY: seeding
device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
# device = 'cpu'

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

if args.num_envs < 12:
    ai_opponents = [microrts_ai.coacAI for _ in range(args.num_bot_envs)]
else:
    ai_opponents = [microrts_ai.coacAI for _ in range(args.num_bot_envs - 6)] + \
                   [microrts_ai.randomBiasedAI for _ in range(2)] + \
                   [microrts_ai.lightRushAI for _ in range(2)] + \
                   [microrts_ai.workerRushAI for _ in range(2)]
ai_opponent_names = [ai.__name__ for ai in ai_opponents]

envs = MicroRTSGridModeVecEnv(
    num_selfplay_envs=args.num_selfplay_envs,
    num_bot_envs=args.num_bot_envs,
    max_steps=2000,
    render_theme=2,
    ai2s=ai_opponents,
    map_path="maps/8x8/basesWorkers8x8.xml",
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
)
envs = MicroRTSStatsRecorder(envs, args.gamma)
envs = VecMonitor(envs)
if args.capture_video:
    envs = VecVideoRecorder(envs, f'videos/{experiment_name}',
                            record_video_trigger=lambda x: x % 1000000 == 0, video_length=2000)
# if args.prod_mode:
#     envs = VecPyTorch(
#         SubprocVecEnv([make_env(args.gym_id, args.seed+i, i) for i in range(args.num_envs)], "fork"),
#         device
#     )
assert isinstance(envs.action_space, MultiDiscrete), "only MultiDiscrete action space is supported"


# ALGO LOGIC: initialize agent here:
class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[], sw=None):
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
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.).to(device))
        return -p_log_p.sum(-1)


class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


class Transpose(nn.Module):
    def __init__(self, permutation):
        super().__init__()
        self.permutation = permutation

    def forward(self, x):
        return x.permute(self.permutation)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Critic(nn.Module):
    def __init__(self, embed_size, map_size):
        super(Critic, self).__init__()
        self.embed_size = embed_size
        self.map_size = map_size
        self.unit_action_space = envs.action_space.nvec[1:].sum()
        self.decoder = layer_init(nn.Linear(self.embed_size, 1), std=1)

    # x: network's output: [seqlength(V), batch_dim, embed_size]
    # player_unit_counts: [batch_dim]
    # (V is the max number of units across all envs).
    # output: [num_envs]
    # TODO study effects of summation vs. averaging vs. weighting both
    # TODO study effects of using only player vs all outputs.
    def forward(self, x, player_unit_counts, observation_mask):
        x_reshaped = x.permute((1, 0, 2))  # [V, N, embed_size] -> [N, V, embed_size]
        value_preds = torch.squeeze(self.decoder(x_reshaped))
        out = torch.sum(value_preds.masked_fill(observation_mask, 0), axis=1)
        # # Iterating over game envs
        # for i, _ in enumerate(x_reshaped):
        #     # Assume that the player 1's positions are output first
        #     value_pred = self.decoder(x_reshaped[i, :player_unit_counts[i], :])
        #     out[i] = torch.sum(value_pred)

        return out


class Actor(nn.Module):
    def __init__(self, embed_size, map_size):
        super(Actor, self).__init__()
        self.embed_size = embed_size
        self.map_size = map_size
        self.unit_action_space = envs.action_space.nvec[1:].sum()
        self.decoder = layer_init(nn.Linear(embed_size, self.unit_action_space), std=0.01)

    # x: network's output: [V, num_envs, embed_size]
    # (V is the max number of units across all envs).
    # player_unit_positions: [batch_dim, height * width]
    # player_unit_counts: [batch_dim]
    # output: [num_envs, env_height * env_width * unit_action_params]
    def forward(self, x, player_unit_positions, player_unit_counts):
        x_reshaped = x.permute((1, 0, 2))  # [V, N, embed_size] -> [N, V, embed_size]
        N = x_reshaped.shape[0]
        out = torch.zeros(N, self.map_size, self.unit_action_space).to(device)
        # Iterating over game envs
        action_preds = self.decoder(x_reshaped[:, :torch.max(player_unit_counts), :])
        for i, _ in enumerate(x_reshaped):
            # Assume that the player 1's positions are output first
            # action_pred = self.decoder(x_reshaped[i, :player_unit_counts[i], :])
            unit_indices = player_unit_positions[i].nonzero()
            out[i, unit_indices[:, 0]] = action_preds[i, :player_unit_counts[i], :]
            # for j, pos in enumerate(player_unit_positions[i]):
            #     out[i, pos, :] = action_pred[j]

        y = out.view(N, self.map_size * self.unit_action_space)
        return y


class Agent(nn.Module):
    def __init__(self, mapsize=8 * 8):
        super(Agent, self).__init__()
        self.mapsize = mapsize
        # For our case 8,8,27
        # The first part of the embedding contains the location of the unit on the map
        # The rest are one-hot encodings of the observation features:
        # hit points, resources, owner, unit types, current action
        # On an 8 x 8 map this is 91
        # TODO by fixing this embed size we are limiting model parameter size.
        # TODO I should consider a compacter format that generates an embedding.
        self.embed_size = (mapsize + 5 + 5 + 3 + 8 + 6)
        # Number of encoding layers
        self.num_layers = 3
        # Picked for divisibility reasons. Will see if there is more wiggle-room here later.
        self.num_heads = 7
        # Ignore dropout for now
        self.dropout = 0
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_size,
                                                   nhead=self.num_heads,
                                                   dim_feedforward=512)
        self.network = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        print(f'Main NN params: {sum([p.numel() for p in self.network.parameters()])}')
        print(f'Trainable params: {sum([p.numel() for p in self.network.parameters() if p.requires_grad])}')
        self.actor = Actor(self.embed_size, self.mapsize)
        self.critic = Critic(self.embed_size, self.mapsize)

    # TODO Type this stuff
    def forward(self, x_reshaped, bool_mask):
        # For some alien reason PyTorch breaks all convention and wants the batch dimension 2nd.
        # [batch_dim, seq_length(V), obs_state] -> # [seq_length(V), batch_dim, obs_state]
        return self.network(x_reshaped.permute(1, 0, 2), src_key_padding_mask=bool_mask)

    def get_action(self, x,
                   observation_mask,
                   player_unit_count,
                   player_unit_position,
                   action=None, invalid_action_masks=None, envs=None, ):
        # x_reshaped, bool_mask, player_unit_pos, player_unit_counts = self.reshape_for_transformer(x)
        # There's no point in passing in tensors we will mask out anyway, so we trim them here.
        max_units_in_batch = torch.max(player_unit_count)
        trimmed_x = x[:, :max_units_in_batch, :]
        trimmed_mask = observation_mask[:, :max_units_in_batch]
        logits = self.actor(self.forward(trimmed_x, trimmed_mask), player_unit_position, player_unit_count)

        # OLD CODE - DO NOT CHANGE
        grid_logits = logits.view(-1, envs.action_space.nvec[1:].sum())
        split_logits = torch.split(grid_logits, envs.action_space.nvec[1:].tolist(), dim=1)

        if action is None:
            invalid_action_masks = torch.tensor(np.array(envs.vec_client.getMasks(0))).to(device)
            invalid_action_masks = invalid_action_masks.view(-1, invalid_action_masks.shape[-1])
            split_invalid_action_masks = torch.split(invalid_action_masks[:, 1:], envs.action_space.nvec[1:].tolist(),
                                                     dim=1)
            multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in
                                  zip(split_logits, split_invalid_action_masks)]
            action = torch.stack([categorical.sample() for categorical in multi_categoricals])
        else:
            invalid_action_masks = invalid_action_masks.view(-1, invalid_action_masks.shape[-1])
            action = action.view(-1, action.shape[-1]).T
            split_invalid_action_masks = torch.split(invalid_action_masks[:, 1:], envs.action_space.nvec[1:].tolist(),
                                                     dim=1)
            multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in
                                  zip(split_logits, split_invalid_action_masks)]
        logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        num_predicted_parameters = len(envs.action_space.nvec) - 1
        logprob = logprob.T.view(-1, self.mapsize, num_predicted_parameters)
        entropy = entropy.T.view(-1, self.mapsize, num_predicted_parameters)
        action = action.T.view(-1, self.mapsize, num_predicted_parameters)
        invalid_action_masks = invalid_action_masks.view(-1, self.mapsize, envs.action_space.nvec[1:].sum() + 1)
        return action, logprob.sum(1).sum(1), entropy.sum(1).sum(1), invalid_action_masks

    def get_value(self, x, observation_mask, player_unit_count):
        # x_reshaped, bool_mask, _, player_unit_counts = self.reshape_for_transformer(x)
        max_units_in_batch = torch.max(player_unit_count)
        trimmed_x = x[:, :max_units_in_batch, :]
        trimmed_mask = observation_mask[:, :max_units_in_batch]
        return self.critic(self.forward(trimmed_x, trimmed_mask), player_unit_count, trimmed_mask)


mapsize = 8 * 8
# Used for transformers
observation_size = (mapsize + 5 + 5 + 3 + 8 + 6)
agent = Agent(mapsize).to(device)
optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
if args.anneal_lr:
    # https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/defaults.py#L20
    lr = lambda f: f * args.learning_rate

# ALGO Logic: Storage for epoch data
action_space_shape = (mapsize, envs.action_space.shape[0] - 1)
observation_shape = envs.observation_space.shape
invalid_action_shape = (mapsize, envs.action_space.nvec[1:].sum() + 1)

obs = torch.zeros(
    (args.num_steps, args.num_envs, observation_shape[0] * observation_shape[1], observation_size)).to(device)
obs_mask = torch.ones((args.num_steps, args.num_envs, observation_shape[0] * observation_shape[1]),
                      dtype=torch.bool).to(device)
unit_counts = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long).to(device)
unit_positions = torch.zeros((args.num_steps, args.num_envs, observation_shape[0] * observation_shape[1])).to(device)

actions = torch.zeros((args.num_steps, args.num_envs) + action_space_shape).to(device)
logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
values = torch.zeros((args.num_steps, args.num_envs)).to(device)
invalid_action_masks = torch.zeros((args.num_steps, args.num_envs) + invalid_action_shape).to(device)
# TRY NOT TO MODIFY: start the game
global_step = 0
start_time = time.time()
# Note how `next_obs` and `next_done` are used; their usage is equivalent to
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/84a7582477fb0d5c82ad6d850fe476829dddd2e1/a2c_ppo_acktr/storage.py#L60
next_obs, next_mask, next_unit_count, next_unit_position = reshape_observation(torch.Tensor(envs.reset()).to(device),
                                                                               device)
next_done = torch.zeros(args.num_envs).to(device)
num_updates = args.total_timesteps // args.batch_size

## CRASH AND RESUME LOGIC:
starting_update = 1
from jpype.types import JArray, JInt

if args.prod_mode and wandb.run.resumed:
    starting_update = run.summary.get('charts/update') + 1
    global_step = starting_update * args.batch_size
    api = wandb.Api()
    run = api.run(f"{run.entity}/{run.project}/{run.id}")
    model = run.file('agent.pt')
    model.download(f"models/{experiment_name}/")
    agent.load_state_dict(torch.load(f"models/{experiment_name}/agent.pt", map_location=device))
    agent.eval()
    print(f"resumed at update {starting_update}")

for update in range(starting_update, num_updates + 1):
    # Annealing the rate if instructed to do so.
    if args.anneal_lr:
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = lr(frac)
        optimizer.param_groups[0]['lr'] = lrnow

    # TRY NOT TO MODIFY: prepare the execution of the game.
    print('Playing...')
    for step in range(0, args.num_steps):
        envs.render()
        global_step += 1 * args.num_envs
        obs[step] = next_obs
        obs_mask[step] = next_mask
        unit_counts[step] = next_unit_count
        unit_positions[step] = next_unit_position
        dones[step] = next_done
        # ALGO LOGIC: put action logic here
        with torch.no_grad():
            values[step] = agent.get_value(obs[step],
                                           obs_mask[step],
                                           unit_counts[step]).flatten()
            action, logproba, _, invalid_action_masks[step] = agent.get_action(obs[step],
                                                                               obs_mask[step],
                                                                               unit_counts[step],
                                                                               unit_positions[step],
                                                                               envs=envs)

        actions[step] = action
        logprobs[step] = logproba

        # TRY NOT TO MODIFY: execute the game and log data.
        # the real action adds the source units
        real_action = torch.cat([
            torch.stack(
                [torch.arange(0, mapsize, device=device) for i in range(envs.num_envs)
                 ]).unsqueeze(2), action], 2)

        # at this point, the `real_action` has shape (num_envs, map_height*map_width, 8)
        # so as to predict an action for each cell in the map; this obviously include a
        # lot of invalid actions at cells for which no source units exist, so the rest of
        # the code removes these invalid actions to speed things up
        real_action = real_action.cpu().numpy()
        valid_actions = real_action[invalid_action_masks[step][:, :, 0].bool().cpu().numpy()]
        valid_actions_counts = invalid_action_masks[step][:, :, 0].sum(1).long().cpu().numpy()
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
            raw_obs, rs, ds, infos = envs.step(java_valid_actions)
            next_obs, next_mask, next_unit_count, next_unit_position = reshape_observation(
                torch.Tensor(raw_obs).to(device), device)
        except Exception as e:
            e.printStackTrace()
            raise
        rewards[step], next_done = torch.Tensor(rs).to(device), torch.Tensor(ds).to(device)

        for i, info in enumerate(infos):
            # When an episode ends there will be breakdown of reward by category of shape:
            # {'WinLossRewardFunction': -1.0, 'ResourceGatherRewardFunction': 8.0, 'ProduceWorkerRewardFunction': 7.0,
            # 'ProduceBuildingRewardFunction': 0.0, 'AttackRewardFunction': 3.0, 'ProduceCombatUnitRewardFunction': 0.0}
            if 'episode' in info.keys():
                print(f"global_step={global_step}, episode_reward={info['episode']['r']}")
                writer.add_scalar("charts/episode_reward", info['episode']['r'], global_step)
                for key in info['microrts_stats']:
                    writer.add_scalar(f"charts/episode_reward/{key}", info['microrts_stats'][key], global_step)
                # Add win-loss reward specific to ai opponent:
                writer.add_scalar(f"charts/episode_reward/WinLossRewardFunction/{ai_opponent_names[i]}",
                                  info['microrts_stats']['WinLossRewardFunction'], global_step)
                break

    # bootstrap reward if not done. reached the batch limit
    # print('Calculating Advantage...')
    with torch.no_grad():
        last_value = agent.get_value(next_obs.to(device),
                                     next_mask.to(device),
                                     next_unit_count.to(device)).reshape(1, -1)
        if args.gae:
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = last_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
        else:
            returns = torch.zeros_like(rewards).to(device)
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    next_return = last_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    next_return = returns[t + 1]
                returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
            advantages = returns - values

    # flatten the batch
    b_obs = obs.reshape(obs.shape[0] * obs.shape[1], obs.shape[2], obs.shape[3])
    b_obs_mask = obs_mask.reshape(obs_mask.shape[0] * obs_mask.shape[1], obs_mask.shape[2])
    b_unit_counts = unit_counts.flatten()
    b_unit_positions = unit_positions.reshape(unit_positions.shape[0] * unit_positions.shape[1],
                                              unit_positions.shape[2])
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + action_space_shape)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)
    b_invalid_action_masks = invalid_action_masks.reshape((-1,) + invalid_action_shape)

    # Optimising the policy and value network
    inds = np.arange(args.batch_size, )
    print('Running PPO...')
    for i_epoch_pi in tqdm.tqdm(range(args.update_epochs)):
        np.random.shuffle(inds)
        for start in range(0, args.batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            minibatch_ind = inds[start:end]
            mb_advantages = b_advantages[minibatch_ind]
            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
            # raise
            _, newlogproba, entropy, _ = agent.get_action(
                b_obs[minibatch_ind],
                b_obs_mask[minibatch_ind],
                b_unit_counts[minibatch_ind],
                b_unit_positions[minibatch_ind],
                b_actions.long()[minibatch_ind],
                b_invalid_action_masks[minibatch_ind],
                envs)
            ratio = (newlogproba - b_logprobs[minibatch_ind]).exp()

            # Stats
            approx_kl = (b_logprobs[minibatch_ind] - newlogproba).mean()

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            entropy_loss = entropy.mean()

            # Value loss
            new_values = agent.get_value(b_obs[minibatch_ind],
                                         b_obs_mask[minibatch_ind],
                                         b_unit_counts[minibatch_ind]).view(-1)
            if args.clip_vloss:
                v_loss_unclipped = ((new_values - b_returns[minibatch_ind]) ** 2)
                v_clipped = b_values[minibatch_ind] + torch.clamp(new_values - b_values[minibatch_ind], -args.clip_coef,
                                                                  args.clip_coef)
                v_loss_clipped = (v_clipped - b_returns[minibatch_ind]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((new_values - b_returns[minibatch_ind]) ** 2)

            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

    ## CRASH AND RESUME LOGIC:
    if args.prod_mode:
        if not os.path.exists(f"models/{experiment_name}"):
            os.makedirs(f"models/{experiment_name}")
            torch.save(agent.state_dict(), f"{wandb.run.dir}/agent.pt")
            wandb.save(f"agent.pt")
        else:
            if update % CHECKPOINT_FREQUENCY == 0:
                torch.save(agent.state_dict(), f"{wandb.run.dir}/agent.pt")

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]['lr'], global_step)
    writer.add_scalar("charts/update", update, global_step)
    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropy.mean().item(), global_step)
    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    if args.kle_stop or args.kle_rollback:
        writer.add_scalar("debug/pg_stop_iter", i_epoch_pi, global_step)
    writer.add_scalar("charts/sps", int(global_step / (time.time() - start_time)), global_step)
    print("Steps per sec:", int(global_step / (time.time() - start_time)))

envs.close()
writer.close()
