import argparse
import pickle

from rlkit.core import logger
from rlkit.samplers.rollout_functions import multitask_rollout
from rlkit.torch import pytorch_util as ptu
from rlkit.envs.vae_wrapper import VAEWrappedEnv
import torch
import gym
import multiworld
from multiworld.core.image_env import ImageEnv
from multiworld.envs.mujoco.cameras import *


def simulate_policy(args):
    data = torch.load(open(args.file, "rb"))
    policy = data['evaluation/policy']
    env = data['evaluation/env']
    imsize = 48
   # multiworld.register_all_envs()
    #env_multi = gym.make('SawyerPushNIPS-v0')
   # env_multi = gym.make('SawyerMultiObj-v0')
  #  env_multi = ImageEnv(
  #      env_multi,
  #      imsize = imsize,
  #      init_camera=sawyer_pusher_camera_upright_v1,
  #      transpose=True,
   #     normalize=True,
  #  )
    #env.wrapped_env = env_multi
    print("Policy and environment loaded")
    imsize = 48
    #env_multi = gym.make('SawyerMultiObj-v0')
#    env.wrapped_env.set_env(env_multi)
#    env._wrapped_env._env = env_multi
 #   env._wrapped_env = env_multi
#    print(env._wrapped_env)
#    print(env._wrapped_env._wrapped_env)
#    print(env._wrapped_env)
#
#    env = ImageEnv(
#        env_multi,
#        imsize = imsize,
#        init_camera=sawyer_pusher_camera_upright_v1,
#        transpose=True,
#        normalize=True,
#    )
#    env = VAEWrappedEnv(env, vae=data['vae'], sample_from_true_prior=True, imsize=48)
    #if args.gpu:
    ptu.set_gpu_mode(True)
    #policy.to(ptu.device)
    if isinstance(env, VAEWrappedEnv) and hasattr(env, 'mode'):
        env.mode(args.mode)
    #if args.enable_render or hasattr(env, 'enable_render'):
        # some environments need to be reconfigured for visualization



    env.enable_render()
    paths = []
    env._goal_sampling_mode = 'env'
    while True:
        paths.append(multitask_rollout(
            env,
            policy,
            max_path_length=args.H,
            render=False,
            observation_key='latent_observation',
            desired_goal_key='latent_desired_goal',
        ))
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics(paths)
        if hasattr(env, "get_diagnostics"):
            for k, v in env.get_diagnostics(paths).items():
                logger.record_tabular(k, v)
        logger.dump_tabular()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=10,
                        help='Speedup')
    parser.add_argument('--mode', default='video_env', type=str,
                        help='env mode')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--enable_render', action='store_true')
    parser.add_argument('--hide', action='store_true')
    args = parser.parse_args()

    simulate_policy(args)
