# Configurations

ROS has to be installed

Install Turtlebot3
https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/

Install Gazebo
https://emanual.robotis.com/docs/en/platform/turtlebot3/simulation/#gazebo-simulation

```
Install packages
sudo apt-get install ros-noetic-gazebo*
sudo apt-get install ros-noetic-cmake*
sudo apt-get install ros-noetic-tf2-sensor-msgs

Install Catkins
cd ~/catkin_ws && catkin_make

```


# Usage
```
roslaunch drl_agent trainDQN.launch
roslaunch drl_agent trainA2C.launch
roslaunch drl_agent trainQL.launch

roslaunch drl_agent evalDQN.launch
roslaunch drl_agent evalA2C.launch
roslaunch drl_agent evalQL.launch
```

## Useful commands

```
killall -9 gzserver gzclient
```

```
rqt_multiplot
```

You have to load the configuration that is in ```config/rqt_multiplot.xml```

# Useful Routes

To modify simulation speed:
```~/catkin_ws/src/turtlebot3_simulations/turtlebot3_gazebo/worlds```
Normal:
```
<real_time_update_rate>1000.0</real_time_update_rate>
<max_step_size>0.001</max_step_size>
```

20x:
```
<real_time_update_rate>0</real_time_update_rate>
<max_step_size>0.003</max_step_size>
```

Activate/Deactivate UI
```
~/catkin_ws/src/turtlebot3_simulations/turtlebot3_gazebo/launch
```
```
<arg name="gui" value="true"/>
```

Using the console you can restart the map (also the topic):
```
rostopic pub -1 /syscommand std_msgs/String "data: 'reset'"
```

# Useful links
https://github.com/DLR-RM/stable-baselines3/tree/master/stable_baselines3/common
https://stable-baselines3.readthedocs.io/en/master/index.html
http://joschu.net/docs/nuts-and-bolts.pdf
http://wiki.ros.org/ROS/Tutorials/UnderstandingTopics
http://wiki.ros.org/openai_ros
https://gazebosim.org/tutorials?tut=physics_params&cat=physics
