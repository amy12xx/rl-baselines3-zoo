import argparse
import glob
import importlib
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch as th
import yaml
import gym
import cv2
from gym import spaces
from stable_baselines3.common.utils import set_random_seed
# from stable_baselines3.common.vec_env import VecFrameStack
import utils.import_envs  # noqa: F401 pylint: disable=unused-import
from utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams
from utils.exp_manager import ExperimentManager
from utils.utils import StoreDict


def compute_stacking(
        num_envs: int,
        n_stack: int,
        render_dim: int,
        ) -> Tuple[bool, int, np.ndarray, int]:
        channels_first = False

        # This includes the vec-env dimension (first)
        stack_dimension = 1 if channels_first else -1
        repeat_axis = 0 if channels_first else -1
        low = np.repeat(np.zeros((render_dim, render_dim, 3)), n_stack, axis=repeat_axis)
        stackedobs = np.zeros((num_envs,) + low.shape, low.dtype)
        return channels_first, stack_dimension, stackedobs, repeat_axis


def update_stacked_obs(observations, stackedobs, stack_dimension=-1):
    stack_ax_size = observations.shape[stack_dimension]
    stackedobs = np.roll(stackedobs, shift=-stack_ax_size, axis=stack_dimension)
    stackedobs[..., -observations.shape[stack_dimension]:] = observations

    return stackedobs


def get_img(env, render_dim):
    try:
        img_obs = env.render("rgb_array", width=render_dim, height=render_dim)
    except:
        img_obs = env.render("rgb_array")
        img_obs = cv2.resize(img_obs, (render_dim, render_dim), interpolation=cv2.INTER_CUBIC)
    return img_obs

def main():  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="environment ID", type=str, default="CartPole-v1")
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="rl-trained-agents")
    parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=1000, type=int)
    parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
    parser.add_argument("--n-envs", help="number of environments", default=1, type=int)
    parser.add_argument("--exp-id", help="Experiment ID (default: 0: latest, -1: no exp folder)", default=0, type=int)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument(
        "--no-render", action="store_true", default=False, help="Do not render the environment (useful for tests)"
    )
    parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions")
    parser.add_argument(
        "--load-best", action="store_true", default=False, help="Load best model instead of last model if available"
    )
    parser.add_argument(
        "--load-checkpoint",
        type=int,
        help="Load checkpoint instead of last model if available, "
        "you must pass the number of timesteps corresponding to it",
    )
    parser.add_argument(
        "--load-last-checkpoint",
        action="store_true",
        default=False,
        help="Load last checkpoint instead of last model if available",
    )
    parser.add_argument("--stochastic", action="store_true", default=False, help="Use stochastic actions")
    parser.add_argument(
        "--norm-reward", action="store_true", default=False, help="Normalize reward if applicable (trained with VecNormalize)"
    )
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument("--reward-log", help="Where to log reward", default="", type=str)
    parser.add_argument(
        "--gym-packages",
        type=str,
        nargs="+",
        default=[],
        help="Additional external Gym environment package modules to import (e.g. gym_minigrid)",
    )
    parser.add_argument(
        "--env-kwargs", type=str, nargs="+", action=StoreDict, help="Optional keyword argument to pass to the env constructor"
    )
    parser.add_argument("--save-episodes", action="store_true", default=False, help="Save episodes")
    parser.add_argument("--reward-threshold", help="Reward threshold", type=int, default=0)
    parser.add_argument("--img-obs", help="Save observations as images", action="store_true", default=False)
    parser.add_argument("--frame-stack", help="Frame stacking", type=int, default=4)
    parser.add_argument("--render-dim", help="Image dimensions", type=int, default=None)
    parser.add_argument("--dense-reward", help="If dense reward, done flag will not be checked while saving trajectories", action="store_true", default=False)
    args = parser.parse_args()

    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_id = args.env
    algo = args.algo
    folder = args.folder

    if args.exp_id == 0:
        args.exp_id = get_latest_run_id(os.path.join(folder, algo), env_id)
        print(f"Loading latest experiment, id={args.exp_id}")

    # Sanity checks
    if args.exp_id > 0:
        log_path = os.path.join(folder, algo, f"{env_id}_{args.exp_id}")
    else:
        log_path = os.path.join(folder, algo)

    assert os.path.isdir(log_path), f"The {log_path} folder was not found"

    found = False
    for ext in ["zip"]:
        model_path = os.path.join(log_path, f"{env_id}.{ext}")
        found = os.path.isfile(model_path)
        if found:
            break

    if args.load_best:
        model_path = os.path.join(log_path, "best_model.zip")
        found = os.path.isfile(model_path)

    if args.load_checkpoint is not None:
        model_path = os.path.join(log_path, f"rl_model_{args.load_checkpoint}_steps.zip")
        found = os.path.isfile(model_path)

    if args.load_last_checkpoint:
        checkpoints = glob.glob(os.path.join(log_path, "rl_model_*_steps.zip"))
        if len(checkpoints) == 0:
            raise ValueError(f"No checkpoint found for {algo} on {env_id}, path: {log_path}")

        def step_count(checkpoint_path: str) -> int:
            # path follow the pattern "rl_model_*_steps.zip", we count from the back to ignore any other _ in the path
            return int(checkpoint_path.split("_")[-2])

        checkpoints = sorted(checkpoints, key=step_count)
        model_path = checkpoints[-1]
        found = True

    if not found:
        raise ValueError(f"No model found for {algo} on {env_id}, path: {model_path}")

    print(f"Loading {model_path}")

    # Off-policy algorithm only support one env for now
    off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]

    if algo in off_policy_algos:
        args.n_envs = 1

    set_random_seed(args.seed)

    if args.num_threads > 0:
        if args.verbose > 1:
            print(f"Setting torch.num_threads to {args.num_threads}")
        th.set_num_threads(args.num_threads)

    is_atari = ExperimentManager.is_atari(env_id)
    is_robotics = ExperimentManager.is_robotics_env(env_id)
    print("is_atari={}, is_robotics={}".format(is_atari, is_robotics))

    stats_path = os.path.join(log_path, env_id)
    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)

    # load env_kwargs if existing
    env_kwargs = {}
    args_path = os.path.join(log_path, env_id, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path, "r") as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]
    # overwrite with command line arguments
    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)

    log_dir = args.reward_log if args.reward_log != "" else None

    env = create_test_env(
        env_id,
        n_envs=args.n_envs,
        stats_path=stats_path,
        seed=args.seed,
        log_dir=log_dir,
        should_render=not args.no_render,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )

    kwargs = dict(seed=args.seed)
    if algo in off_policy_algos:
        # Dummy buffer size as we don't need memory to enjoy the trained agent
        kwargs.update(dict(buffer_size=1))

    # Check if we are running python 3.8+
    # we need to patch saved model under python 3.6/3.7 to load them
    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

    custom_objects = {}
    if newer_python_version:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }

    model = ALGOS[algo].load(model_path, env=env, custom_objects=custom_objects, **kwargs)

    save_episode_obs, save_episode_acts = [], []
    save_episode_stackedobs = []
    ep_obs, ep_acts = [], []

    channels_first, stack_dimension, stackedobs, repeat_axis = compute_stacking(1, args.frame_stack, args.render_dim)
    ep_stacked_obs = []

    obs = env.reset()

    # to save observations from Fetch env
    if args.img_obs:
        if is_atari:
            ep_obs.append(obs)
            stackedobs = update_stacked_obs(obs, stackedobs)
            ep_stacked_obs.append(stackedobs)
        elif args.render_dim is not None:
            img_obs = get_img(env, args.render_dim)
            img_obs = img_obs.astype(np.uint8)
            ep_obs.append(img_obs)
            stackedobs = update_stacked_obs(img_obs, stackedobs)
            ep_stacked_obs.append(stackedobs)
        else:
            img_obs = env.render("rgb_array")
            ep_obs.append(img_obs)
            stackedobs = update_stacked_obs(img_obs, stackedobs)
            ep_stacked_obs.append(stackedobs)
    else:
        ep_obs.append(obs["observation"])
        stackedobs = update_stacked_obs(obs["observation"], stackedobs)
        ep_stacked_obs.append(stackedobs)

    # Deterministic by default except for atari games
    stochastic = args.stochastic or is_atari and not args.deterministic
    deterministic = not stochastic

    state = None
    episode_reward = 0.0
    episode_rewards, episode_lengths = [], []
    ep_len = 0
    # For HER, monitor success rate
    successes = []

    print("Running for {} timesteps".format(args.n_timesteps))
    try:
        for _ in range(args.n_timesteps):
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            obs, reward, done, infos = env.step(action)
            ep_acts.append(action)

            if not args.no_render:
                env.render("human")

            episode_reward += reward[0]
            ep_len += 1

            print("rew {}, len {}, done {}".format(episode_reward, ep_len, done))

            if args.n_envs == 1:
                # For atari the return reward is not the atari score
                # so we have to get it from the infos dict
                if is_atari and infos is not None and args.verbose >= 1:
                    episode_infos = infos[0].get("episode")
                    if episode_infos is not None:
                        print(f"Atari Episode Score: {episode_infos['r']:.2f}")
                        print("Atari Episode Length", episode_infos["l"])
                if done:
                    print("Episode reward: {}".format(episode_reward))
                    if episode_reward >= args.reward_threshold:
                        assert len(ep_obs) == len(ep_acts), "len not same: {}, {}".format(len(ep_obs), len(ep_acts))
                        save_episode_obs.append(ep_obs)
                        save_episode_acts.append(ep_acts)
                        save_episode_stackedobs.append(ep_stacked_obs)
                    ep_obs, ep_acts = [], []
                    ep_stacked_obs = []
                    obs = env.reset()
                elif args.dense_reward:
                    if episode_reward >= args.reward_threshold:
                        print("Episode reward: {}".format(episode_reward))
                        save_episode_obs.append(ep_obs)
                        save_episode_acts.append(ep_acts)
                        save_episode_stackedobs.append(ep_stacked_obs)
                    ep_obs, ep_acts = [], []
                    ep_stacked_obs = []
                if args.img_obs:
                    # img_obs = env.render("rgb_array")
                    if is_atari:
                        ep_obs.append(obs)
                        stackedobs = update_stacked_obs(obs, stackedobs)
                        ep_stacked_obs.append(stackedobs)
                    elif args.render_dim is not None:
                        img_obs = get_img(env, args.render_dim)
                        img_obs = img_obs.astype(np.uint8)
                        ep_obs.append(img_obs)
                        stackedobs = update_stacked_obs(img_obs, stackedobs)
                        ep_stacked_obs.append(stackedobs)
                    else:
                        img_obs = env.render("rgb_array")
                        ep_obs.append(img_obs)
                        stackedobs = update_stacked_obs(img_obs, stackedobs)
                        ep_stacked_obs.append(stackedobs)
                else:
                    ep_obs.append(obs["observation"])
                    stackedobs = update_stacked_obs(obs["observation"], stackedobs)
                    ep_stacked_obs.append(stackedobs)

                if done and not is_atari and args.verbose > 0:
                    # NOTE: for env using VecNormalize, the mean reward
                    # is a normalized reward when `--norm_reward` flag is passed
                    print(f"Episode Reward: {episode_reward:.2f}")
                    print("Episode Length", ep_len)
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(ep_len)
                    episode_reward = 0.0
                    ep_len = 0
                    state = None

                # Reset also when the goal is achieved when using HER
                if done and infos[0].get("is_success") is not None:
                    if args.verbose > 1:
                        print("Success?", infos[0].get("is_success", False))

                    if infos[0].get("is_success") is not None:
                        successes.append(infos[0].get("is_success", False))
                        episode_reward, ep_len = 0.0, 0

    # except KeyboardInterrupt:
    except Exception as e:
        print("ERROR: ", e)
        pass

    if args.verbose > 0 and len(successes) > 0:
        print(f"Success rate: {100 * np.mean(successes):.2f}%")

    if args.verbose > 0 and len(episode_rewards) > 0:
        print(f"{len(episode_rewards)} Episodes")
        print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")

    if args.verbose > 0 and len(episode_lengths) > 0:
        print(f"Mean episode length: {np.mean(episode_lengths):.2f} +/- {np.std(episode_lengths):.2f}")

    env.close()

    #save episodes
    print("Saving episodes...")
    save_episode_obs = [obs for ep in save_episode_obs for obs in ep]
    save_episode_acts = [act for ep in save_episode_acts for act in ep]
    obs = np.array(save_episode_obs)
    acts = np.array(save_episode_acts)

    save_episode_stackedobs = [obs for ep in save_episode_stackedobs for obs in ep]
    stackedobs = np.array(save_episode_stackedobs)
    stackedobs = np.squeeze(stackedobs, axis=1)
    print(stackedobs.shape)

    if args.img_obs:
        # episodes = obs
        episodes = stackedobs
    else:
        obs = obs.reshape(len(obs), -1)
        acts = acts.reshape(len(acts), -1)
        episodes = np.hstack((obs, acts))
    print("Episode shape: ", episodes.shape)
    np.save("{}/expert_{}".format(log_path, args.env), episodes)


if __name__ == "__main__":
    main()
