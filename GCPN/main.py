# Import packages
from mpi4py import MPI
from baselines.common import set_global_seeds
from baselines import logger
from tensorboardX import SummaryWriter
import os
import tensorflow as tf
from argparse import Namespace
import gym
from gym_molecule.envs.molecule import GraphEnv

# set_memory_growth() allocates exclusively the GPU memory needed
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
if len(physical_devices) is not 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def train(args, seed, writer=None):
    from baselines.ppo1 import pposgd_simple_gcn, gcn_policy
    import baselines.common.tf_util as U
    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()
    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    if args.env == 'molecule':
        env = gym.make('molecule-v0')
        env.init(data_type=args.dataset, logp_ratio=args.logp_ratio, qed_ratio=args.qed_ratio, sa_ratio=args.sa_ratio,
                 reward_step_total=args.reward_step_total, is_normalize=args.normalize_adj,
                 reward_type=args.reward_type, reward_target=args.reward_target, has_feature=bool(args.has_feature),
                 is_conditional=bool(args.is_conditional), conditional=args.conditional, max_action=args.max_action,
                 min_action=args.min_action)
    elif args.env == 'graph':
        env = GraphEnv()
        env.init(reward_step_total=args.reward_step_total, is_normalize=args.normalize_adj, dataset=args.dataset)
    print(env.observation_space)

    def policy_fn(name, ob_space, ac_space):
        return gcn_policy.GCNPolicy(name=name, ob_space=ob_space, ac_space=ac_space, atom_type_num=env.atom_type_num,
                                    args=args)

    env.seed(workerseed)
    pposgd_simple_gcn.learn(args, env, policy_fn, max_timesteps=args.num_steps, timesteps_per_actorbatch=256,
                            clip_param=0.2, entcoeff=0.01, optim_epochs=8, optim_stepsize=args.lr, optim_batchsize=32,
                            gamma=1, lam=0.95, schedule='linear', writer=writer)
    env.close()


if __name__ == '__main__':
    # List of arguments
    args = {'bn': 0, 'conditional': 'low', 'curriculum': 0, 'curriculum_num': 6, 'curriculum_step': 200,
            'dataset': 'zinc', 'dataset_load': 'zinc', 'emb_size': 128, 'env': 'molecule', 'expert_end': 1000,
            'expert_start': 0, 'gan_final_ratio': 1, 'gan_step_ratio': 1, 'gan_type': 'normal', 'gate_sum_d': 0,
            'gcn_aggregate': 'mean', 'graph_emb': 0, 'has_concat': 0, 'has_d_final': 1, 'has_d_step': 1,
            'has_feature': 0, 'has_ppo': 1, 'has_residual': 0, 'is_conditional': 0, 'layer_num_d': 3, 'layer_num_g': 3,
            'load': 0, 'load_step': 250, 'logp_ratio': 1, 'lr': 0.001, 'mask_null': 0, 'max_action': 128,
            'min_action': 20, 'name': 'test_conditional', 'name_full': '', 'name_full_load': '',
            'name_load': '0new_concatno_mean_layer3_expert1500', 'normalize_adj': 0, 'num_steps': 500,
            'qed_ratio': 1, 'reward_step_total': 0.5, 'reward_target': 0.5, 'reward_type': 'logppen', 'rl_end': 200,
            'rl_start': 250, 'sa_ratio': 1, 'save_every': 50, 'seed': 666, 'stop_shift': -3, 'supervise_time': 4}
    args = Namespace(**args)
    print(args)
    args.name_full = args.env + '_' + args.dataset + '_' + args.name
    args.name_full_load = args.env + '_' + args.dataset_load + '_' + args.name_load + '_' + str(args.load_step)
    # check and clean
    if not os.path.exists('molecule_gen'):
        os.makedirs('molecule_gen')
    if not os.path.exists('ckpt'):
        os.makedirs('ckpt')

    # only keep first worker result in tensorboard
    if MPI.COMM_WORLD.Get_rank() == 0:
        writer = SummaryWriter(comment='_' + args.dataset + '_' + args.name)
    else:
        writer = None
    train(args, seed=args.seed, writer=writer)

# Arguments description and default values
# argument('--dataset', type=str, default='zinc', help='caveman; grid; ba; zinc; gdb')
# argument('--reward_type', type=str, default='logppen',
#                     help='logppen;logp_target;qed;qedsa;qed_target;mw_target;gan')
# argument('--gcn_aggregate', type=str, default='mean', help='sum, mean, concat')
# argument('--gan_type', type=str, default='normal', help='normal, recommend, wgan')
#
# args = {'bn': 0, 'conditional': 'low', 'curriculum': 0, 'curriculum_num': 6, 'curriculum_step': 200,
#         'dataset': 'zinc', 'dataset_load': 'zinc', 'emb_size': 128, 'env': 'molecule', 'expert_end': 1000000,
#         'expert_start': 0, 'gan_final_ratio': 1, 'gan_step_ratio': 1, 'gan_type': 'normal', 'gate_sum_d': 0,
#         'gcn_aggregate': 'mean', 'graph_emb': 0, 'has_concat': 0, 'has_d_final': 1, 'has_d_step': 1,
#         'has_feature': 0, 'has_ppo': 1, 'has_residual': 0, 'is_conditional': 0, 'layer_num_d': 3, 'layer_num_g': 3,
#         'load': 0, 'load_step': 250, 'logp_ratio': 1, 'lr': 0.001, 'mask_null': 0, 'max_action': 128,
#         'min_action': 20, 'name': 'test_conditional', 'name_full': '', 'name_full_load': '',
#         'name_load': '0new_concatno_mean_layer3_expert1500', 'normalize_adj': 0, 'num_steps': 50000000,
#         'qed_ratio': 1, 'reward_step_total': 0.5, 'reward_target': 0.5, 'reward_type': 'logppen', 'rl_end': 1000000,
#         'rl_start': 250, 'sa_ratio': 1, 'save_every': 200, 'seed': 666, 'stop_shift': -3, 'supervise_time': 4}
