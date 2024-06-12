#!/usr/bin/env python3

import time
import numpy as np
import rospkg
# ROS packages required
import rospy
from gym import wrappers
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
from stable_baselines3 import PPO


def main():
    loadModel = False
    saveModel = True
    continueTraining = False
    env, modelPath = init()

    if (loadModel):
        rospy.logwarn("Loading Model...")
        model = PPO.load(modelPath)
        inited = False
    else:
        if (continueTraining):
            rospy.logwarn("Continue training")
            model = PPO.load(modelPath, env)
        else:
            model = PPO('MultiInputPolicy', env, verbose=1)
      
        model.learn(total_timesteps=1000000)
        rospy.logwarn("Training finished")
        inited = True

        if (saveModel):
            rospy.logwarn("Saving Model...")
            model.save(modelPath)
            rospy.logwarn("Model saved")
        

    rospy.logwarn("Start prediction...")
    evaluate(model, env, inited)


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
                obs = env.getObs()
                inited = False
                rospy.logwarn(str(obs))
            else:
                obs = env.reset()
                
                
            done = False
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_rewards.append(reward)

            all_episode_rewards.append(sum(episode_rewards))

        mean_episode_reward = np.mean(all_episode_rewards)
        rospy.logwarn("Mean reward: " + str(mean_episode_reward) + " Num episodes: " + str(num_episodes))

        return mean_episode_reward

def init():
    rospy.init_node('train_gppo',
                    anonymous=True, log_level=rospy.WARN)
    task_and_robot_environment_name = rospy.get_param(
        '/turtlebot3/task_and_robot_environment_name')
    env = StartOpenAI_ROS_Environment(
        task_and_robot_environment_name)

    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('drl_agent')
    outdir = pkg_path + '/training_results'
    modelPath = outdir + "/GDDPG"
    env = wrappers.Monitor(env, outdir, force=True)
    return env, modelPath

if __name__ == '__main__':
    main()
