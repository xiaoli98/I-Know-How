from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_util import make_vec_env

import gymnasium as gym
from gymnasium.wrappers.record_video import RecordVideo
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.sac.sac import SACMaster
from stable_baselines3.sac.sac_master import SACMasterPolicy

from typing import Callable

import os
import pickle

import argparse

import wandb
from wandb.integration.sb3 import WandbCallback


parser = argparse.ArgumentParser('stable baseline tests on highway Env')

parser.add_argument('--env', default="highway-fast-v0", type=str, help='environment to be trained on')
parser.add_argument('--ncpu', default=16, type=int, help="number of cpu to use")
parser.add_argument('--training', default=True, type=bool, help="if train the algorithm")
parser.add_argument('--run-name', default="default", type=str, help="run name")
parser.add_argument('--project-name', default="sb3_base_models", type=str, help="name of the wandb project")
parser.add_argument('--vehicle-number', default=5, type=int, help="non controlled vehicle appearing in during simulation")
parser.add_argument('--random_spawn', default=False, type=bool, help="make the vehicle spawn in random point of the track")
parser.add_argument('--ntags', nargs='+')
parser.add_argument('--subpolicies', default="MIRIN", type=str)

parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--epoch', default=1, type=int)
parser.add_argument('--step-per-epoch', default=10_000_000, type=int)
parser.add_argument('--batch-size', default=1024, type=int)
parser.add_argument('--gradient-steps', default=10, type=int)
parser.add_argument('--tau', default=0.9, type=float)
parser.add_argument('--gamma', default=0.99, type=float)
parser.add_argument('--ent-coef', default="auto_0.5", type=str)
parser.add_argument('--learning-starts', default="50_000", type=int)
parser.add_argument('--weighting-scheme', default="classic", type=str)

args = parser.parse_args()

KINEMATICS_OBSERVATION = {
    "type": "Kinematics",
    "vehicles_count": 5,
    "features": ["presence", "x", "y", "vx", "vy", "heading", "long_off", "lat_off", "ang_off"],
    "absolute": False,
    "order": "sorted",
}

config = {
    "observation": KINEMATICS_OBSERVATION,
    "action": {
        "type": "ContinuousAction",
    },
    "policy_frequency": 15, 
    "vehicles_count": args.vehicle_number,
    "random_spawn": args.random_spawn,
    }

env_name = args.env
TRAINING = args.training
N_CPU = args.ncpu
PROJECT_NAME = args.project_name
TAGS = args.ntags
SUBPOLICIES = args.subpolicies

EPOCHS = args.epoch
STEP_PER_EPOCH = args.step_per_epoch
BATCH_SIZE = args.batch_size
LR = args.lr
GRADIENT_STEPS=args.gradient_steps
TAU=args.tau
GAMMA=args.gamma
ENT_COEF=args.ent_coef
LEARNING_STARTS=args.learning_starts
WEIGTH_SCHEME = args.weighting_scheme

NAME = f"sac_master_{env_name}"
LOGNAME = f"{NAME}"

SAVE_PATH = f"./checkpoint/{NAME}/"

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
    print(f"created directories {SAVE_PATH}")

TOTAL_STEPS = EPOCHS * STEP_PER_EPOCH

policy_kwargs = dict(net_arch=dict(pi=[512, 512, 512], qf=[512, 512, 512]))

def main():
    # only for tacking configuration
    env = gym.make(env_name, config=config)
    run = wandb.init(
        project=PROJECT_NAME,
        name=NAME if args.run_name == "default" else args.run_name,
        config={**config, 
                **vars(args), 
                "env_config": env.unwrapped.config,
                },
        tags=TAGS,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )
    env.close()
    
    WEIGHTS_LOG_PATH = f"log_weight/{run.id}"
    if not os.path.exists(WEIGHTS_LOG_PATH):
        os.makedirs(WEIGHTS_LOG_PATH)
        print(f"created directories {WEIGHTS_LOG_PATH}")
    
    if TRAINING:
        env = make_vec_env(env_name, n_envs=N_CPU, env_kwargs={"config": config}, vec_env_cls=SubprocVecEnv)

        model = SACMaster(SACMasterPolicy, env, 
                learning_rate=LR,
                learning_starts=LEARNING_STARTS,
                buffer_size=1_000_000,
                batch_size=BATCH_SIZE,
                gradient_steps=GRADIENT_STEPS,
                # policy_kwargs=policy_kwargs,
                tau=TAU, 
                gamma=GAMMA,
                ent_coef=ENT_COEF,
                train_freq= 128,
                action_noise=OrnsteinUhlenbeckActionNoise(mean=np.zeros(1), sigma=0.1*np.ones(1), dtype=np.float32),
                tensorboard_log=f"log/{NAME}/{run.id}",
                verbose=1,
                sub_policies_path=f"./src/checkpoint/subpolicies_5vehicles_{SUBPOLICIES}/",
                weighting_scheme=WEIGTH_SCHEME,
                # eta = (1, 0.01, 500_000),
                # trainable_subpolicy=False,
                )

        for epoch in range(0, EPOCHS):
            model.learn(total_timesteps=STEP_PER_EPOCH, 
                        log_interval=10,
                        reset_num_timesteps=False, 
                        progress_bar=True,
                        tb_log_name=LOGNAME,
                        callback=WandbCallback(
                                model_save_path=f"{SAVE_PATH}{run.id}",
                                verbose=0,
                        ))
            test_env = gym.make(env_name, config=config, render_mode="rgb_array")
            test_env = RecordVideo(test_env, video_folder=f"./videos/{NAME}/",
                            episode_trigger=lambda x: True,
                            disable_logger= True)
            # wrapper for better video
            # test_env.unwrapped.set_record_video_wrapper(test_env)
            weights_to_save = []
            for video in range(30):
                done = truncated = False
                weigths_per_episodes= []
                obs, info = test_env.reset()
                while not (done or truncated):
                    obs = np.expand_dims(obs, axis=0)
                    action, weights, _states = model.predict(obs, deterministic=True)
                    weigths_per_episodes.append([weights])
                    obs, reward, done, truncated, info = test_env.step(action[0])
                weights_to_save.append(weigths_per_episodes)
            with open(f"{WEIGHTS_LOG_PATH}/test_weights_epoch_{epoch}.pth", "wb") as f:
                pickle.dump(weights_to_save, f)
            test_env.close()

        print("training finished")
        env.close()
    
if __name__ == "__main__":
    main()