import argparse
import os
import sys
from distutils.util import strtobool


def get_run_args():
    parser = argparse.ArgumentParser(description='PPO agent')

    subparsers = parser.add_subparsers(help='sub-command help', dest='command')

    parser_resume = subparsers.add_parser('resume', help='when you want to resume training an existing agent.')
    parser_resume.add_argument('--run-id', type=str, required=True, default=None,
                               help='supply the wandb ID of the run to resume.')
    parser_resume.add_argument('--wandb-project-name', type=str, default='cleanRL',
                               help="the wandb's project name")

    parser_new = subparsers.add_parser('new', help='when you want to train a new agent.')
    # Common arguments
    parser_new.add_argument('--exp-name', type=str, required=True,
                            help='the name of this experiment')
    parser_new.add_argument('--gym-id', type=str, default="MicrortsDefeatCoacAIShaped-v3",
                            help='the id of the gym environment')
    parser_new.add_argument('--learning-rate', type=float, default=2.5e-4,
                            help='the learning rate of the optimizer')
    parser_new.add_argument('--seed', type=int, default=1,
                            help='seed of the experiment')
    parser_new.add_argument('--total-timesteps', type=int, default=100000000,
                            help='total timesteps of the experiments')
    parser_new.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs='?',
                            const=True,
                            help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser_new.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                            help='if toggled, cuda will not be enabled by default')
    parser_new.add_argument('--prod-mode', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                            help='run the script in production mode and use wandb to log outputs')
    parser_new.add_argument('--capture-video', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                            help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser_new.add_argument('--wandb-project-name', type=str, default="cleanRL",
                            help="the wandb's project name")
    parser_new.add_argument('--wandb-entity', type=str, default=None,
                            help="the entity (team) of wandb's project")

    # Algorithm specific arguments
    parser_new.add_argument('--n-minibatch', type=int, default=4,
                            help='the number of mini batch')
    parser_new.add_argument('--num-bot-envs', type=int, default=24,
                            help='the number of bot game environment; 16 bot envs means 16 games')
    parser_new.add_argument('--num-selfplay-envs', type=int, default=0,
                            help='the number of self play envs; 16 self play envs means 8 games')
    parser_new.add_argument('--num-steps', type=int, default=256,
                            help='the number of steps per game environment')
    parser_new.add_argument('--gamma', type=float, default=0.99,
                            help='the discount factor gamma')
    parser_new.add_argument('--gae-lambda', type=float, default=0.95,
                            help='the lambda for the general advantage estimation')
    parser_new.add_argument('--ent-coef', type=float, default=0.01,
                            help="coefficient of the entropy")
    parser_new.add_argument('--vf-coef', type=float, default=0.5,
                            help="coefficient of the value function")
    parser_new.add_argument('--max-grad-norm', type=float, default=0.5,
                            help='the maximum norm for the gradient clipping')
    parser_new.add_argument('--clip-coef', type=float, default=0.1,
                            help="the surrogate clipping coefficient")
    parser_new.add_argument('--update-epochs', type=int, default=4,
                            help="the K epochs to update the policy")
    parser_new.add_argument('--kle-stop', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                            help='If toggled, the policy updates will be early stopped w.r.t target-kl')
    parser_new.add_argument('--kle-rollback', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                            help='If toggled, the policy updates will roll back to previous policy if KL exceeds target-kl')
    parser_new.add_argument('--target-kl', type=float, default=0.03,
                            help='the target-kl variable that is referred by --kl')
    parser_new.add_argument('--gae', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                            help='Use GAE for advantage computation')
    parser_new.add_argument('--norm-adv', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                            help="Toggles advantages normalization")
    parser_new.add_argument('--anneal-lr', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                            help="Toggle learning rate annealing for policy and value networks")
    parser_new.add_argument('--clip-vloss', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                            help='Toggles wheter or not to use a clipped loss for the value function, as per the paper.')
    parser_new.add_argument('--sparse-rewards', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                            help='Toggle to enable whether sparse rewards should be used instead of shaped ones.')
    parser_new.add_argument('--transformer-layers', type=int, default=5,
                            help='the number of layers to use in the transformer encoder')
    parser_new.add_argument('--feed-forward-neurons', type=int, default=512,
                            help='the number of feed-forward neurons in a transformer layer')
    parser_new.add_argument('--reward-weights', type=float, nargs=6,
                            help="string of reward weights. Needs to be exactly 6 weights. For example: "
                                 "10.0, 1.0, 1.0, 0.2, 1.0, 4.0")
    parser_new.add_argument('--embed-factor', type=int, default=5,
                            help='how large to make the embedding per symbol')
    parser_new.add_argument('--map-size', type=int, default=8,
                            help='which size of map to use. Currently supported 8, 16')
    parser_new.add_argument('--attention-heads', type=int, default=7,
                            help='determines number of heads to use for multi-headed attention in transformer')
    parser_new.add_argument('--input-padding', type=int, default=0,
                            help='how much to pad inputs to the transformer encoder by (can use this to balance out '
                                 'heads)')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        print('Please provide one of the available sub-commands above.')
        sys.exit()

    return args
