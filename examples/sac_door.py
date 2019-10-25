from gym.envs.mujoco import HalfCheetahEnv
from multiworld.envs.mujoco.sawyer_xyz import sawyer_door
from multiworld.core.image_env import ImageEnv
from rlkit.envs import mujoco_image_env

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, CNNTanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.conv_networks import CNN, CNNPolicy

from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from torch import nn as nn
import torch
from cnn_specs import cnn_specs

def experiment(variant):
    imsize = 48
    expl_env = mujoco_image_env.ImageMujocoEnv(sawyer_door.SawyerDoorEnv(), imsize=imsize)
    eval_env =  mujoco_image_env.ImageMujocoEnv(sawyer_door.SawyerDoorEnv(), imsize=imsize)

    expl_env.reset()
    eval_env.reset()
    print(expl_env)
    #expl_env = ImageEnv(sawyer_door_hook_env())
    #eval_env = ImageEnv(sawyer_door_hook_env())
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size
    M = variant['layer_size']

    cnn_args = cnn_specs[1]
    print(cnn_args)

    qf1 = CNN(input_width=imsize, input_height=imsize, input_channels=3, output_size=1, \
    kernel_sizes=cnn_args["kernel_sizes"], strides=cnn_args["strides"], paddings =cnn_args["paddings"], \
    hidden_sizes=cnn_args["hidden_sizes"], n_channels=cnn_args["n_channels"], added_fc_input_size=action_dim)

    qf2 = CNN(input_width=imsize, input_height=imsize, input_channels=3, output_size=1, \
    kernel_sizes=cnn_args["kernel_sizes"], strides=cnn_args["strides"],  paddings=cnn_args["paddings"], \
    hidden_sizes=cnn_args["hidden_sizes"], n_channels=cnn_args["n_channels"], added_fc_input_size=action_dim)

    target_qf1 = CNN(input_width=imsize, input_height=imsize, input_channels=3, output_size=1, \
    kernel_sizes=cnn_args["kernel_sizes"], strides=cnn_args["strides"],  paddings=cnn_args["paddings"], \
    hidden_sizes=cnn_args["hidden_sizes"], n_channels=cnn_args["n_channels"], added_fc_input_size=action_dim)

    target_qf2 = CNN(input_width=imsize, input_height=imsize, input_channels=3, output_size=1, \
    kernel_sizes=cnn_args["kernel_sizes"], strides=cnn_args["strides"],  paddings=cnn_args["paddings"], \
    hidden_sizes=cnn_args["hidden_sizes"], n_channels=cnn_args["n_channels"], added_fc_input_size=action_dim)

    policy = CNNTanhGaussianPolicy(input_width=imsize, input_height=imsize, input_channels=3, action_dim=action_dim, \
    kernel_sizes=cnn_args["kernel_sizes"], strides=cnn_args["strides"], paddings=cnn_args["paddings"], \
    hidden_sizes=cnn_args["hidden_sizes"], n_channels=cnn_args["n_channels"])

    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
    )

    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )

    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()




if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1E5),
        algorithm_kwargs=dict(
            num_epochs=3000,
            num_eval_steps_per_epoch=1000,
            num_trains_per_train_loop=500,
            num_expl_steps_per_train_loop=5000,
            min_num_steps_before_training=5000,
            max_path_length=50,
            batch_size=64,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
    )
    print(torch.backends.cudnn.enabled)

    setup_logger('name-of-experiment', variant=variant)
    ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant)
