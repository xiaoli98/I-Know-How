from copy import deepcopy
from multiprocessing import Process
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
from stable_baselines3.sac.pnn import PNN_Policy

import torch as th

import os
from tqdm import tqdm

import argparse

import wandb
from wandb.integration.sb3 import WandbCallback


parser = argparse.ArgumentParser('stable baseline tests on highway Env')

parser.add_argument('--env', default="highway-fast-v0", type=str, help='environment to be trained on')
parser.add_argument('--resume', default=False, type=bool, help="if resume from some checkpoint")
parser.add_argument('--ncpu', default=4, type=int, help="number of cpu to use")
parser.add_argument('--training', default=True, type=bool, help="if train the algorithm")
parser.add_argument('--project-name', default="sb3_base_models", type=str, help="name of the wandb project")
parser.add_argument('--vehicle-number', default=10, type=int, help="non controlled vehicle appearing in during simulation")

parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--epoch', default=1, type=int)
parser.add_argument('--step-per-epoch', default=100_000, type=int)
parser.add_argument('--batch-size', default=64, type=int)
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
    "vehicles_count": 5,
    }

env_name = "racetrack-v0"
RESUME = args.resume
TRAINING = args.training
N_CPU = args.ncpu
PROJECT_NAME = args.project_name

EPOCHS = args.epoch
STEP_PER_EPOCH = args.step_per_epoch
BATCH_SIZE = args.batch_size
LR = args.lr
GRADIENT_STEPS=args.gradient_steps
TAU=args.tau
GAMMA=args.gamma
ENT_COEF=args.ent_coef

NAME = None

# sac_master_racetrack-v0
# sac_racetrack-v0
# sac_pnn_racetrack-v0

skills= "MIRIN"
PROJECT = f"sac_master_racetrack-v0"
to_evaluate_path = f"./src/evaluation/{PROJECT}"
# sub_policy_path = f"./src/evaluation/ablation/policy_{skills}.pth"

def evaluate_f(run_id, seed):
    print(f"evaluating: {run_id}")
    NAME = f"{PROJECT}"
    env = gym.make(env_name, config=config)
    run = wandb.init(
        project="ablation_skills",
        name=NAME,
        config={**config, 
                # **vars(args), 
                "env_config": env.unwrapped.config,
                },
        tags=[PROJECT, f"seed={seed}"],
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        # monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )
    env.close()
    # env = make_vec_env(env_name, n_envs=N_CPU, env_kwargs={"config": config}, vec_env_cls=SubprocVecEnv)

    ############################################################
    # toggle the comment based on the model you are evaluating #
    ############################################################
    
    # evaluating classical SAC
    # model = SAC.load(f"{to_evaluate_path}/{run_id}/model.zip")
    
    # for SAC with PNN and SAC master, the subpolicies should be preloaded by instatiating the object
    # then load the policy
    
    # evaluating SAC with PNN structure
    # model = SAC(PNN_Policy, env, 
    #         learning_rate=LR,
    #         buffer_size=1_000_000,
    #         batch_size=BATCH_SIZE,
    #         gradient_steps= args.gradient_steps,
    #         tau=TAU, # (1-tau)*target + tau * source 
    #         gamma=GAMMA,
    #         ent_coef=ENT_COEF,
    #         train_freq= 15,
    #         action_noise=OrnsteinUhlenbeckActionNoise(mean=np.zeros(1), sigma=0.1*np.ones(1), dtype=np.float32),
    #         tensorboard_log=f"log/{NAME}/{run.id}",
    #         verbose=1,
    #         )
    
    # evaluating SAC master
    model = SACMaster(SACMasterPolicy, env, 
            learning_rate=LR,
            learning_starts=0,
            buffer_size=1_000_000,
            batch_size=BATCH_SIZE,
            gradient_steps=10,
            tau=0.9, # (1-tau)*target + tau * source 
            gamma=0.99,
            ent_coef="auto_0.5",
            train_freq= 15,
            action_noise=OrnsteinUhlenbeckActionNoise(mean=np.zeros(1), sigma=0.1*np.zeros(1), dtype=np.float32),
            tensorboard_log=f"log/{NAME}",
            verbose=1,
            sub_policies_path=f"./src/checkpoint/subpolicies_5vehicles_{skills}/",
            )
    model.policy.load_state_dict(th.load(f"{to_evaluate_path}/{run_id}/policy.pth", map_location=th.device('cpu')))
    # model.policy.load_state_dict(th.load(f"{sub_policy_path}", map_location=th.device('cpu')))

    test_env = gym.make(env_name, config=config, render_mode="rgb_array")
    
    
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

    section_passed = [0]*10
    
    section_passed_start_from = {
        ('a', 'b'): [0,0],
        ('b', 'c'): [0,0],
        ('c', 'd'): [0,0],
        ('d', 'e'): [0,0],
        ('e', 'f'): [0,0],
        ('f', 'g'): [0,0],
        ('g', 'h'): [0,0],
        ('h', 'i'): [0,0],
        ('i', 'a'): [0,0],
    }
    
    section_activated_skill = {
        ('a', 'b'): np.zeros(4),
        ('b', 'c'): np.zeros(4),
        ('c', 'd'): np.zeros(4),
        ('d', 'e'): np.zeros(4),
        ('e', 'f'): np.zeros(4),
        ('f', 'g'): np.zeros(4),
        ('g', 'h'): np.zeros(4),
        ('h', 'i'): np.zeros(4),
        ('i', 'a'): np.zeros(4),
    }
    section_staying_counter = {
            ('a', 'b'): np.zeros(1),
            ('b', 'c'): np.zeros(1),
            ('c', 'd'): np.zeros(1),
            ('d', 'e'): np.zeros(1),
            ('e', 'f'): np.zeros(1),
            ('f', 'g'): np.zeros(1),
            ('g', 'h'): np.zeros(1),
            ('h', 'i'): np.zeros(1),
            ('i', 'a'): np.zeros(1),
        }
    # obs, info = test_env.reset()
    for video in tqdm(range(100)):
        done = truncated = False
        obs, info = test_env.reset(seed=seed+video)
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
        
        spawining_lane = current_lane
        
        while not (done or truncated):
            action, weights, _states = model.predict(obs)
            section_staying_counter[current_lane] += 1
            section_activated_skill[current_lane] += np.array(weights)
            
            # action, _states = model.predict(obs)
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
        section_passed_start_from[spawining_lane][0] += passed_sections
        section_passed_start_from[spawining_lane][1] += 1
        if passed_sections < 10:
            section_passed[passed_sections] += 1
        else:
            section_passed[-1] += 1
        # print(f"section passed: {section_passed}")
        # print(f"section passed spawning: {section_passed_start_from}")
    for k, v in section_activated_skill.items():
        v /= section_staying_counter[k]
    wandb.log({
        "section activated skill": str(section_activated_skill),
        # "mean section passed": np.mean(all_passed_sections),
        # "mean distance per action": np.mean(all_mean_dist_per_action),
        # "mean action per section": np.mean(per_sector_actions),
        # "section passed": section_passed,
        # "section passed start from": str(section_passed_start_from)
    })
    test_env.close()

    print(f"{run_id} evaluated")
    # model.save(f"{SAVE_PATH}final.pth")
    # model.save_replay_buffer(f"{SAVE_PATH}replay.pth")
    env.close()
    wandb.finish()
    del model


if __name__ == "__main__":
    run_ids = os.listdir(to_evaluate_path)
    processes = []
    print(f"{run_ids}\n total of :{len(run_ids)}")
    
    seeds = [0, 100, 200, 300, 400]
    for seed in seeds:
        for run_id in run_ids:
            # evaluate_f(run_id)
            p = Process(target=evaluate_f, args=[run_id, seed, ])
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()