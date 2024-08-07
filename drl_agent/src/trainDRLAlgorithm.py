#!/usr/bin/env python3

import time
import numpy as np
import rospkg
# ROS packages required
import rospy
from gymnasium import wrappers
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
from stable_baselines3 import DQN, PPO
from sb3_contrib import QRDQN
from dict_mini_resnet import DictMinimalResNet, DictImageNet, DictImageNet5Channel

import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import CallbackList,CheckpointCallback

def main():
    loadModel = rospy.get_param('/turtlebot3/load_model')
    continueTraining = rospy.get_param('/turtlebot3/continueTraining')
    saveModel = rospy.get_param('/turtlebot3/saveModel')
    use_wandb = rospy.get_param('/turtlebot3/use_wandb')
    architecture = rospy.get_param('/turtlebot3/config')
    
    config = {
        "algorithm": rospy.get_param('/turtlebot3/algorithm'),
        "policy_type": rospy.get_param('/turtlebot3/policy_type'),
        "total_timesteps": rospy.get_param('/turtlebot3/total_timesteps'),
        "n_steps_before_every_PPO_update": rospy.get_param('/turtlebot3/n_steps_before_every_PPO_update')
    }
    
    run = None
    if use_wandb:
        # Set up W&B
        run = wandb.init(
            project="DLLab",
            config=config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            # monitor_gym=True,  # auto-upload the videos of agents playing the game
            # save_code=True,  # optional
        )

    env, modelPath = init(config["algorithm"])
    
    if loadModel:
        rospy.logwarn("Loading Model...")
        model = loadModelfunc(config["algorithm"], modelPath + rospy.get_param('/turtlebot3/load_model_path'), env)
        inited = False
    else:
        if continueTraining:
            rospy.logwarn("Continue training")
            model = loadModelfunc(config["algorithm"], modelPath + rospy.get_param('/turtlebot3/load_model_path'), env)
        else:
            model = startModel(config["algorithm"], env, run, config, rospy.get_param('/turtlebot3/use_resnet'), architecture)
        
        if use_wandb:
            checkpoint_callback = CheckpointCallback(save_freq=20000, save_path=modelPath,
                                         name_prefix='rl_model')
            wandb_callback = WandbCallback(
                            model_save_path=f"{modelPath}/{run.id}",
                            verbose=2,
                        )
            list_callback = CallbackList([checkpoint_callback,wandb_callback])
            model.learn(total_timesteps=config["total_timesteps"],
                        callback=list_callback,
            )
            run.finish()
            rospy.logwarn("Training finished")
        else:
            model.learn(total_timesteps=config["total_timesteps"])
            
            rospy.logwarn("Training finished")

            if (saveModel):
                rospy.logwarn("Saving Model...")
                model.save(modelPath + "/model")
                rospy.logwarn("Model saved")
                
        inited = True
    
    if rospy.get_param('/turtlebot3/evaluate'):
        rospy.logwarn("Start prediction...")
        evaluate(model, env, inited)

def loadModelfunc(algorithm, modelPath, env = None):
    if algorithm == "DQN":
        return DQN.load(modelPath, env=env)
    elif algorithm =="PPO":
        return PPO.load(modelPath, env=env)
    elif algorithm=="DDQN":
        return QRDQN.load(modelPath, env=env)
    else:
        rospy.logwarn("No valid algorihtm!")
        return None
    
def startModel(algorithm, env, run, config, use_resnet, architecture):
    if use_resnet:
        if architecture == "DictImageNet":
            policy_kwargs = dict(
                features_extractor_class=DictImageNet,
                features_extractor_kwargs=dict(features_dim=32),
            )
        elif architecture == "DictImageNet5Channel":
            policy_kwargs = dict(
                features_extractor_class=DictImageNet5Channel,
                features_extractor_kwargs=dict(features_dim=32),
            )
    else:
        policy_kwargs = {}

    if algorithm == "DQN":
        learning_rate = 0.00100
        buffer_size = 50000
        batch_size = 64
        gamma = 0.99
        train_freq = (200, "step")
        if run:
            return DQN(config["policy_type"], env, learning_rate=learning_rate, buffer_size=buffer_size, batch_size=batch_size, gamma=gamma, train_freq = train_freq, verbose=1, tensorboard_log=f"runs/{run.id}", policy_kwargs=policy_kwargs)
        else:
            return DQN(config["policy_type"], env, learning_rate=learning_rate, buffer_size=buffer_size, batch_size=batch_size, gamma=gamma, train_freq = train_freq, verbose=1, policy_kwargs=policy_kwargs)
    elif algorithm =="PPO":
        if run:
            return PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}", n_steps = config["n_steps_before_every_PPO_update"], policy_kwargs=policy_kwargs)
        else:
            return PPO(config["policy_type"], env, verbose=1, n_steps = config["n_steps_before_every_PPO_update"], policy_kwargs=policy_kwargs)
    elif algorithm=="DDQN":
        policy_kwargs["n_quantiles"] = 50
        if run:
            return QRDQN(config["policy_type"], env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log=f"runs/{run.id}", exploration_fraction = 0.1, exploration_final_eps=0.1)
        else:
            return QRDQN(config["policy_type"], env, policy_kwargs=policy_kwargs, verbose=1,  exploration_fraction = 0.1, exploration_final_eps=0.1)
    else:
        rospy.logwarn("No valid algorithm!")
        return None

def evaluate(model, env, inited, num_episodes=10):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    all_episode_rewards = []
    
    for i in range(num_episodes):
        episode_rewards = []

        #Hack needed to enable evaluation post training :/
        if inited:
            obs,info = env.getObs()
            inited = False
            rospy.logwarn(str(obs))
        else:
            obs,info = env.reset()
            
            
        done = False
        ended = False
        while not done and not ended:
            action, _states = model.predict(obs )#,deterministic=True)
            obs, reward, done,ended, info = env.step(action)
            print(f"Reward{reward}")
            print(f"Summed Rewards{sum(episode_rewards)}")
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    rospy.logwarn("Mean reward: " + str(mean_episode_reward) + " Num episodes: " + str(num_episodes))

    return mean_episode_reward

def init(algorithm):
    rospy.init_node('example_turtlebot3_maze_qlearn', anonymous=True, log_level=rospy.WARN)
    task_and_robot_environment_name = rospy.get_param('/turtlebot3/task_and_robot_environment_name')
    env = StartOpenAI_ROS_Environment(task_and_robot_environment_name)

    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('drl_agent')
    outdir = pkg_path + '/training_results'
    modelPath = outdir + fr"/{algorithm}/"
    #env = wrappers.Monitor(env, outdir, force=True)
    return env, modelPath

if __name__ == '__main__':
    main()
