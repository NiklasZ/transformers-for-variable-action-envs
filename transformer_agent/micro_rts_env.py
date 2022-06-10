import time
from typing import Union, Tuple, List

import numpy as np
from gym.spaces import MultiDiscrete
from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
from stable_baselines3.common.vec_env import VecEnvWrapper, VecVideoRecorder


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


def create_envs(args: dict) -> Tuple[Union[VecMonitor, VecVideoRecorder], List[str]]:
    if args['num_envs'] < 12:
        ai_opponents = [microrts_ai.coacAI for _ in range(args['num_bot_envs'])]
    else:
        ai_opponents = [microrts_ai.coacAI for _ in range(args['num_bot_envs'] - 6)] + \
                       [microrts_ai.randomBiasedAI for _ in range(2)] + \
                       [microrts_ai.lightRushAI for _ in range(2)] + \
                       [microrts_ai.workerRushAI for _ in range(2)]
    ai_opponent_names = [ai.__name__ for ai in ai_opponents]

    # Weight order is affects:
    # WinLossRewardFunction(),
    # ResourceGatherRewardFunction(),
    # ProduceWorkerRewardFunction(),
    # ProduceBuildingRewardFunction(),
    # AttackRewardFunction(),
    # ProduceCombatUnitRewardFunction(),

    if 'reward_weights' in args and args['reward_weights'] is not None:
        # Custom weights
        reward_weight = np.array(args['reward_weights'])
    elif args['sparse_rewards']:
        # Only rewards win/lose
        reward_weight = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    else:
        # Rewards other actions like making units and killing others.
        reward_weight = np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])

    if args['map_size'] == 8:
        map_path = "maps/8x8/basesWorkers8x8.xml"
    elif args['map_size'] == 16:
        map_path = "maps/16x16/basesWorkers16x16.xml"
    else:
        raise Exception(f"Unsupported map size {args['map_size']}")

    envs = MicroRTSGridModeVecEnv(
        num_selfplay_envs=args['num_selfplay_envs'],
        num_bot_envs=args['num_bot_envs'],
        max_steps=2000,
        render_theme=2,
        ai2s=ai_opponents,
        map_path=map_path,
        reward_weight=reward_weight
    )
    envs = MicroRTSStatsRecorder(envs, args['gamma'])
    envs = VecMonitor(envs)
    if args['capture_video']:
        envs = VecVideoRecorder(envs, f"videos/{args['experiment_name']}",
                                record_video_trigger=lambda x: x % 1000000 == 0, video_length=2000)

    assert isinstance(envs.action_space, MultiDiscrete), "only MultiDiscrete action space is supported"
    return envs, ai_opponent_names
