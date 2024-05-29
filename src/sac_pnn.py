from copy import deepcopy

from tqdm import tqdm
from stable_baselines3 import SAC
from stable_baselines3.sac.pnn import PNN_Policy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_util import make_vec_env

import gymnasium as gym
from gymnasium.wrappers.record_video import RecordVideo
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

from typing import Callable

import os

import argparse

import wandb
from wandb.integration.sb3 import WandbCallback


parser = argparse.ArgumentParser('stable baseline tests on highway Env')

parser.add_argument('--env', default="highway-fast-v0", type=str, help='environment to be trained on')
parser.add_argument('--ncpu', default=16, type=int, help="number of cpu to use")
parser.add_argument('--training', default=True, type=bool, help="if train the algorithm")
parser.add_argument('--project-name', default="sb3_base_models", type=str, help="name of the wandb project")
parser.add_argument('--vehicle-number', default=5, type=int, help="non controlled vehicle appearing in during simulation")
parser.add_argument('--random_spawn', default=True, type=bool, help="make the vehicle spawn in random point of the track")
parser.add_argument('--ntags', nargs='+')

parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--epoch', default=1, type=int)
parser.add_argument('--step-per-epoch', default=10_000_000, type=int)
parser.add_argument('--batch-size', default=1024, type=int)
parser.add_argument('--gradient-steps', default=10, type=int)
parser.add_argument('--tau', default=0.9, type=float)
parser.add_argument('--gamma', default=0.99, type=float)
parser.add_argument('--ent-coef', default="auto_0.5", type=str)

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
    "policy_frequency": 5, 
    "vehicles_count": args.vehicle_number,
    "random_spawn": args.random_spawn,

    }

env_name = args.env
TRAINING = args.training
N_CPU = args.ncpu
PROJECT_NAME = args.project_name
TAGS = args.ntags

EPOCHS = args.epoch
STEP_PER_EPOCH = args.step_per_epoch
BATCH_SIZE = args.batch_size
LR = args.lr
GRADIENT_STEPS=args.gradient_steps
TAU=args.tau
GAMMA=args.gamma
ENT_COEF=args.ent_coef

NAME = f"sac_pnn_{env_name}"
LOGNAME = f"{NAME}"

SAVE_PATH = f"./checkpoint/{NAME}/"
LOAD_PATH = f"./checkpoint/"

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
    print(f"created directories {SAVE_PATH}")

TOTAL_STEPS = EPOCHS * STEP_PER_EPOCH

def main():
    # only for tacking configuration
    env = gym.make(env_name, config=config)
    run = wandb.init(
        project=PROJECT_NAME,
        name=NAME,
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
    if TRAINING:
        env = make_vec_env(env_name, n_envs=N_CPU, env_kwargs={"config": config}, vec_env_cls=SubprocVecEnv)

        model = SAC(PNN_Policy, env, 
                learning_rate=LR,
                buffer_size=1_000_000,
                batch_size=BATCH_SIZE,
                gradient_steps= args.gradient_steps,
                tau=TAU, # (1-tau)*target + tau * source 
                gamma=GAMMA,
                ent_coef=ENT_COEF,
                train_freq= 15,
                action_noise=OrnsteinUhlenbeckActionNoise(mean=np.zeros(1), sigma=0.1*np.ones(1), dtype=np.float32),
                tensorboard_log=f"log/{NAME}/{run.id}",
                verbose=1,
                )

        for epoch in range(0, EPOCHS):
            model.learn(total_timesteps=STEP_PER_EPOCH, 
                        log_interval=10,
                        reset_num_timesteps=False, 
                        progress_bar=True,
                        tb_log_name=LOGNAME,
                        callback=WandbCallback(
                                model_save_path=f"{SAVE_PATH}{run.id}",
                                model_save_freq=STEP_PER_EPOCH/10,
                                verbose=0,
                        ))

            test_env = gym.make(env_name, config=config, render_mode="rgb_array")
            test_env = RecordVideo(test_env, video_folder=f"./videos/{NAME}/",
                            episode_trigger=lambda x: True,
                            disable_logger= True)
            test_env.unwrapped.set_record_video_wrapper(test_env)
            
            for video in range(10):
                done = truncated = False
                obs, info = test_env.reset()
                while not (done or truncated):
                    action, _stetes = model.predict(obs)
                    obs, reward, done, truncated, info = test_env.step(action)
            test_env.close()

        print("training finished")
        # model.save(f"{SAVE_PATH}final.pth")
        # model.save_replay_buffer(f"{SAVE_PATH}replay.pth")
        env.close()
        wandb.finish()
        return model

def evaluate(model):
    test_env = gym.make(env_name, config=config, render_mode="rgb_array")
    run = wandb.init(
        project="evaluation",
        name=PROJECT_NAME,
        # config={**config, 
        #         # **vars(args), 
        #         "env_config": test_env.unwrapped.config,
        #         },
        tags=[PROJECT_NAME],
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        # monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )
    
    next_lane_dict={
            ('a', 'b'): ('b', 'c'),
            ('b', 'c'): ('c', 'd'),
            ('c', 'd'): ('d', 'e'),
            ('d', 'e'): ('e', 'f'),
            ('e', 'f'): ('f', 'g'),
            ('f', 'g'): ('g', 'h'),
            ('g', 'h'): ('h', 'i'),
            ('h', 'i'): ('i', 'a'),
            ('i', 'a'): ('a', 'b')
        }
    all_passed_sections = []
    all_mean_dist_per_action = []
    per_sector_actions = []
    for video in tqdm(range(100)):
        done = truncated = False
        obs, info = test_env.reset()
        prev_pose = deepcopy(info["pose"])
        road_net = info["road"]
        distance = 0
        episode_length = 0
        passed_sections = 0
        n_actions = 0
        lane_edge_dict = {}
        for key, value in road_net.lanes_dict().items():
            lane_edge_dict[value] = (key[0], key[1])
            
        current_lane = lane_edge_dict[info['current_lane']]
        next_lane = next_lane_dict[current_lane]
        
        while not (done or truncated):
            # action, weights, _states = model.predict(obs)
            action, _states = model.predict(obs)
            obs, reward, done, truncated, info = test_env.step(action)
            distance += np.linalg.norm(prev_pose - info["pose"])
            episode_length += 1
            n_actions += 1
            current_lane = lane_edge_dict[info['current_lane']]
            if current_lane == next_lane:
                next_lane = next_lane_dict[current_lane]
                passed_sections += 1
                per_sector_actions.append(n_actions)
                n_actions = 0
            prev_pose = deepcopy(info["pose"])
        all_passed_sections.append(passed_sections)  
        all_mean_dist_per_action.append(distance/episode_length)          
    
    wandb.log({
        "mean section passed": np.mean(all_passed_sections),
        "mean distance per action": np.mean(all_mean_dist_per_action),
        "mean action per section": np.mean(per_sector_actions)
    })
    test_env.close()
    wandb.finish()
    

if __name__ == "__main__":
    model = main()