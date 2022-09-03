import sys

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import random
import os

from train_agent.mixed_embedded_agent import MixedEmbeddedAgent, reshape_observation_mixed_embedded
from train_agent.arg_handler import get_run_args
from train_agent.micro_rts_env import create_envs
from jpype.types import JArray, JInt
import wandb
from argparse import Namespace

if __name__ == "__main__":
    args = get_run_args()
    resumed = args.command == 'resume'

    if resumed:
        print(f'Resuming run {args.run_id}')
        run = wandb.init(id=args.run_id, project=args.wandb_project_name, resume='must', monitor_gym=True,
                         save_code=True)
        args = Namespace(**run.config.as_dict())
        wandb.tensorboard.patch(save=False)
        experiment_name = f"{args.gym_id}_{args.exp_name}_{args.seed}_{int(time.time())}"
        writer = SummaryWriter(f"/tmp/{experiment_name}")

    elif args.prod_mode:
        run_id = wandb.util.generate_id()
        experiment_name = f'{args.gym_id}_{args.exp_name}_{args.seed}_{int(time.time())}_{run_id}'
        run = wandb.init(id=run_id,
                         project=args.wandb_project_name, entity=args.wandb_entity,
                         config=args, name=experiment_name, monitor_gym=True, save_code=True)
        wandb.tensorboard.patch(save=False)
        writer = SummaryWriter(f"/tmp/{experiment_name}")

    else:
        # TRY NOT TO MODIFY: setup the environment
        experiment_name = f"{args.gym_id}_{args.exp_name}_{args.seed}_{int(time.time())}"
        writer = SummaryWriter(f"runs/{experiment_name}")
        writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
            '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))

    CHECKPOINT_FREQUENCY = 10
    print(f"Running experiment '{experiment_name}'")
    # Using these to track and save the best performing version
    best_reward = -1000000000
    running_reward = 0
    game_count = 0

    if not args.seed:
        args.seed = int(time.time())
args.num_envs = args.num_selfplay_envs + args.num_bot_envs
args.batch_size = int(args.num_envs * args.num_steps)
args.minibatch_size = int(args.batch_size // args.n_minibatch)

# TRY NOT TO MODIFY: seeding
device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
#device = 'cpu'

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

envs, ai_opponent_names = create_envs(vars(args))

mapsize = args.map_size ** 2
# Used for transformers
observation_size = (1 + 5 + 5 + 3 + 8 + 6)  # 27 features + 1 for position
agent = MixedEmbeddedAgent(mapsize, envs, device, args.transformer_layers, args.feed_forward_neurons,
                           args.attention_heads, args.input_padding, args.embed_size).to(device)
agent.network_size()
optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
if args.anneal_lr:
    # https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/defaults.py#L20
    lr = lambda f: f * args.learning_rate

# ALGO Logic: Storage for epoch data
action_space_shape = (mapsize, envs.action_space.shape[0] - 1)
observation_shape = envs.observation_space.shape
invalid_action_shape = (mapsize, envs.action_space.nvec[1:].sum() + 1)

obs = torch.zeros(
    (args.num_steps, args.num_envs, observation_shape[0] * observation_shape[1], observation_size),
    dtype=torch.int16).to(device)
entity_masks = torch.ones((args.num_steps, args.num_envs, observation_shape[0] * observation_shape[1]),
                          dtype=torch.bool).to(device)
entity_counts = torch.zeros((args.num_steps, args.num_envs), dtype=torch.int64).to(device)
unit_positions = torch.zeros((args.num_steps, args.num_envs, observation_shape[0] * observation_shape[1]),
                             dtype=torch.bool).to(device)
unit_masks = torch.ones((args.num_steps, args.num_envs, observation_shape[0] * observation_shape[1]),
                        dtype=torch.bool).to(device)
enemy_unit_masks = torch.ones((args.num_steps, args.num_envs, observation_shape[0] * observation_shape[1]),
                              dtype=torch.bool).to(device)
neutral_unit_masks = torch.ones((args.num_steps, args.num_envs, observation_shape[0] * observation_shape[1]),
                                dtype=torch.bool).to(device)

actions = torch.zeros((args.num_steps, args.num_envs) + action_space_shape, dtype=torch.int16).to(device)
logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
values = torch.zeros((args.num_steps, args.num_envs)).to(device)
invalid_action_masks = torch.zeros((args.num_steps, args.num_envs) + invalid_action_shape, dtype=torch.bool).to(device)
# TRY NOT TO MODIFY: start the game
global_step = 0
start_time = time.time()
# Note how `next_obs` and `next_done` are used; their usage is equivalent to
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/84a7582477fb0d5c82ad6d850fe476829dddd2e1/a2c_ppo_acktr/storage.py#L60
next_obs, next_entity_mask, next_entity_count, next_unit_position, next_unit_mask, next_enemy_unit_mask, next_neutral_unit_mask = \
    reshape_observation_mixed_embedded(torch.Tensor(envs.reset()).to(device), device)
next_done = torch.zeros(args.num_envs).to(device)
num_updates = args.total_timesteps // args.batch_size

## CRASH AND RESUME LOGIC:
starting_update = 1

if args.prod_mode and wandb.run.resumed:
    starting_update = run.summary.get('charts/update') + 1
    global_step = starting_update * args.batch_size
    api = wandb.Api()
    run = api.run(f"{run.entity}/{run.project}/{run.id}")
    model = run.file('agent.pt')
    model.download(f"models/{experiment_name}/")
    opt = run.file('optimizer.pt')
    opt.download(f"models/{experiment_name}/")
    agent.load_state_dict(torch.load(f"models/{experiment_name}/agent.pt", map_location=device))
    agent.eval()
    optimizer.load_state_dict(torch.load(f"models/{experiment_name}/optimizer.pt", map_location=device))
    running_reward = best_reward = run.history()['charts/episode_reward'].dropna().mean()
    print(f"resumed at update {starting_update}")

for update in range(starting_update, num_updates + 1):
    # Clean out of reference tensors
    torch.cuda.empty_cache()

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
        entity_masks[step] = next_entity_mask
        entity_counts[step] = next_entity_count
        unit_positions[step] = next_unit_position
        unit_masks[step] = next_unit_mask
        enemy_unit_masks[step] = next_enemy_unit_mask
        neutral_unit_masks[step] = next_neutral_unit_mask
        dones[step] = next_done
        # ALGO LOGIC: put action logic here
        with torch.no_grad():
            values[step] = agent.get_value(obs[step],
                                           entity_masks[step],
                                           entity_counts[step],
                                           unit_masks[step],
                                           enemy_unit_masks[step],
                                           neutral_unit_masks[step]).flatten()
            action, logproba, _, invalid_action_masks[step] = agent.get_action(obs[step],
                                                                               entity_masks[step],
                                                                               entity_counts[step],
                                                                               unit_positions[step],
                                                                               unit_masks[step],
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

        with torch.no_grad():
            try:
                raw_obs, rs, ds, infos = envs.step(java_valid_actions)
                next_obs, next_entity_mask, next_entity_count, next_unit_position, next_unit_mask, next_enemy_unit_mask, next_neutral_unit_mask = \
                    reshape_observation_mixed_embedded(torch.Tensor(raw_obs).to(device), device)
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
                    win_reward = info['microrts_stats']['WinLossRewardFunction']
                    writer.add_scalar(f"charts/episode_reward/WinLossRewardFunction/{ai_opponent_names[i]}",
                                      win_reward, global_step)
                    if game_count > 0:
                        running_reward += (win_reward - running_reward) / game_count
                    else:
                        running_reward = win_reward
                    break
    # bootstrap reward if not done. reached the batch limit
    # print('Calculating Advantage...')
    with torch.no_grad():
        last_value = agent.get_value(next_obs.to(device),
                                     next_entity_mask.to(device),
                                     next_entity_count.to(device),
                                     next_unit_mask,
                                     next_enemy_unit_mask,
                                     next_neutral_unit_mask).reshape(1, -1)
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
    b_entity_masks = entity_masks.reshape(entity_masks.shape[0] * entity_masks.shape[1], entity_masks.shape[2])
    b_entity_counts = entity_counts.flatten()
    b_unit_positions = unit_positions.reshape(unit_positions.shape[0] * unit_positions.shape[1],
                                              unit_positions.shape[2])
    b_unit_masks = unit_masks.reshape(unit_masks.shape[0] * unit_masks.shape[1], unit_masks.shape[2])
    b_enemy_unit_masks = enemy_unit_masks.reshape(enemy_unit_masks.shape[0] * enemy_unit_masks.shape[1],
                                                  enemy_unit_masks.shape[2])
    b_neutral_unit_masks = neutral_unit_masks.reshape(neutral_unit_masks.shape[0] * neutral_unit_masks.shape[1],
                                                      neutral_unit_masks.shape[2])

    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + action_space_shape)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)
    b_invalid_action_masks = invalid_action_masks.reshape((-1,) + invalid_action_shape)

    torch.cuda.empty_cache()
    # Optimising the policy and value network
    inds = np.arange(args.batch_size, )
    print('Running PPO...\n')
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
                b_entity_masks[minibatch_ind],
                b_entity_counts[minibatch_ind],
                b_unit_positions[minibatch_ind],
                b_unit_masks[minibatch_ind],
                b_actions[minibatch_ind],
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
                                         b_entity_masks[minibatch_ind],
                                         b_entity_counts[minibatch_ind],
                                         b_unit_masks[minibatch_ind],
                                         b_enemy_unit_masks[minibatch_ind],
                                         b_neutral_unit_masks[minibatch_ind]).view(-1)
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

        if update % CHECKPOINT_FREQUENCY == 0:
            torch.save(agent.state_dict(), f"{wandb.run.dir}/agent.pt")
            torch.save(optimizer.state_dict(), f"{wandb.run.dir}/optimizer.pt")
            # torch.save(agent.state_dict(), f"{wandb.run.dir}/agent_{update}.pt")
            # torch.save(optimizer.state_dict(), f"{wandb.run.dir}/optimizer_{update}.pt")

            if best_reward < running_reward:
                best_reward = running_reward
                print(f'Saving new best agent with running return of {best_reward}')
                torch.save(agent.state_dict(), f"{wandb.run.dir}/best_agent.pt")
                torch.save(optimizer.state_dict(), f"{wandb.run.dir}/best_optimizer.pt")
                # torch.save(agent.state_dict(), f"{wandb.run.dir}/best_agent_{update}.pt")
                # torch.save(optimizer.state_dict(), f"{wandb.run.dir}/best_optimizer_{update}.pt")
                # wandb.run.save(f'best_agent_{update}.pt', policy='now')
                # wandb.run.save(f'best_optimizer_{update}.pt', policy='now')

            # wandb.run.save(f'agent_{update}.pt', policy='now')
            # wandb.run.save(f'optimizer_{update}.pt', policy='now')
            wandb.run.save(f'*.pt', policy='now')

            print('Synced agent state to wandb')

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
