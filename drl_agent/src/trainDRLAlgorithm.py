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

import wandb
from wandb.integration.sb3 import WandbCallback


def main():
    loadModel = rospy.get_param('/turtlebot3/load_model')
    continueTraining = rospy.get_param('/turtlebot3/continueTraining')
    saveModel = rospy.get_param('/turtlebot3/saveModel')
    use_wandb = rospy.get_param('/turtlebot3/use_wandb')
    
    config = {
        "algorithm": rospy.get_param('/turtlebot3/algorithm'),
        "policy_type": "MultiInputPolicy",
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

    if (loadModel):
        rospy.logwarn("Loading Model...")
        model = loadModelfunc(config["algorithm"], modelPath + "/mapwithpose100k/model.zip")
        inited = False
    else:
        if (continueTraining):
            rospy.logwarn("Continue training")
            model = loadModelfunc(config["algorithm"], modelPath + "/model")
        else:
            #model = DQN('MlpPolicy', env, verbose=1)
            model = startModel(config["algorithm"], env, run, config)
        
        if use_wandb:
            model.learn(total_timesteps=config["total_timesteps"],
                        callback=WandbCallback(
                            model_save_path=f"{modelPath}/{run.id}",
                            verbose=2,
                        ),
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
        
    # rospy.logwarn("Start prediction...")
    evaluate(model, env, inited)

def loadModelfunc(algorithm, modelPath):
    if algorithm == "DQN":
        return DQN.load(modelPath)
    elif algorithm =="PPO":
        return PPO.load(modelPath)
    elif algorithm=="DDQN":
        return QRDQN.load(modelPath)
    else:
        rospy.logwarn("No valid algorihtm!")
        return None
    
def startModel(algorithm, env, run, config):
    if algorithm == "DQN":
        #TODO use Config Files
        learning_rate = 0.00100
        buffer_size = 50000
        batch_size = 64
        gamma = 0.99
        train_freq = (200, "step")
        if run:
            return DQN(config["policy_type"], env, learning_rate=learning_rate, buffer_size=buffer_size, batch_size=batch_size, gamma=gamma, train_freq = train_freq, verbose=1, tensorboard_log=f"runs/{run.id}")
        else:
            return DQN(config["policy_type"], env, learning_rate=learning_rate, buffer_size=buffer_size, batch_size=batch_size, gamma=gamma, train_freq = train_freq, verbose=1)
    elif algorithm =="PPO":
        if run:
            return PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}", n_steps = config["n_steps_before_every_PPO_update"])
        else:
            return PPO(config["policy_type"], env, verbose=1, n_steps = config["n_steps_before_every_PPO_update"])
    elif algorithm=="DDQN":
        policy_kwargs = dict(n_quantiles=50)
        if run:
            return QRDQN(config["policy_type"], env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log=f"runs/{run.id}")
        else:
            return QRDQN(config["policy_type"], env, policy_kwargs=policy_kwargs, verbose=1)
    else:
        rospy.logwarn("No valid algorihtm!")
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
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done,_, info = env.step(action)
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
