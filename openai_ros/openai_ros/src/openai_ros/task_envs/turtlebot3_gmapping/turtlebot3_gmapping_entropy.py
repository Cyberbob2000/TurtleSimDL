import rospy
import numpy
from gymnasium import spaces
from openai_ros.robot_envs import turtlebot3_env
from gymnasium.envs.registration import register
from geometry_msgs.msg import Vector3
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
from openai_ros.openai_ros_common import ROSLauncher
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header, String, Float64
from gazebo_msgs.msg import PerformanceMetrics
import os
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry
import math
from std_msgs.msg import Float64, Bool
import ros_numpy
#import matplotlib.pyplot as plt

class GmappingTurtleBot3WorldEnv(turtlebot3_env.TurtleBot3Env):
    def __init__(self):
        """
        This Task Env is designed for having the TurtleBot3 in the turtlebot3 world
        closed room with columns.
        It will learn how to move around without crashing.
        """
        # This is the path where the simulation files, the Task and the Robot gits will be downloaded if not there
        ros_ws_abspath = rospy.get_param("/turtlebot3/ros_ws_abspath", None)
        assert ros_ws_abspath is not None, "You forgot to set ros_ws_abspath in your yaml file of your main RL script. Set ros_ws_abspath: \'YOUR/SIM_WS/PATH\'"
        assert os.path.exists(ros_ws_abspath), "The Simulation ROS Workspace path " + ros_ws_abspath + \
                                               " DOESNT exist, execute: mkdir -p " + ros_ws_abspath + \
                                               "/src;cd " + ros_ws_abspath + ";catkin_make"

        ROSLauncher(rospackage_name="turtlebot3_gazebo",
                    launch_file_name="turtlebot3_gmapping.launch",
                    ros_ws_abspath=ros_ws_abspath)

        # Load Params from the desired Yaml file
        LoadYamlFileParamsTest(rospackage_name="openai_ros",
                               rel_path_from_package_to_file="src/openai_ros/task_envs/turtlebot3_gmapping/config",
                               yaml_file_name="turtlebot3_gmapping.yaml")


        # Here we will add any init functions prior to starting the MyRobotEnv
        super(GmappingTurtleBot3WorldEnv, self).__init__(ros_ws_abspath)

        # Only variable needed to be set here
        number_actions = rospy.get_param('/turtlebot3/n_actions')
        self.action_space = spaces.Discrete(number_actions)

        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-numpy.inf, numpy.inf)


        #number_observations = rospy.get_param('/turtlebot3/n_observations')
        """
        We set the Observation space for the 6 observations
        cube_observations = [
            round(current_disk_roll_vel, 0),
            round(y_distance, 1),
            round(roll, 1),
            round(pitch, 1),
            round(y_linear_speed,1),
            round(yaw, 1),
        ]
        """

        # Actions and Observations
        self.linear_forward_speed = rospy.get_param('/turtlebot3/linear_forward_speed')
        self.linear_turn_speed = rospy.get_param('/turtlebot3/linear_turn_speed')
        self.angular_speed = rospy.get_param('/turtlebot3/angular_speed')
        self.init_linear_forward_speed = rospy.get_param('/turtlebot3/init_linear_forward_speed')
        self.init_linear_turn_speed = rospy.get_param('/turtlebot3/init_linear_turn_speed')

        self.new_ranges = rospy.get_param('/turtlebot3/new_ranges')

        self.reset_gmapping = rospy.Publisher('/reset_gmapping', String, queue_size=10)

        # We create two arrays based on the binary values that will be assigned
        # In the discretization method.
        laser_scan = self.get_laser_scan()
        self.min_range = rospy.get_param('/turtlebot3/min_range')
        self.max_laser_value = laser_scan.range_max
        self.min_laser_value = rospy.get_param('/turtlebot3/min_laser_value')
    
        num_laser_readings = 5
        rospy.logwarn("MALDITASEA " + str(num_laser_readings))
        high = numpy.full((num_laser_readings), 1)
        low = numpy.full((num_laser_readings), self.min_laser_value / self.max_laser_value)

        high_coverage = numpy.full((1), 1)
        low_coverage = numpy.full((1), 0)
        # We only use two integers
        
        #self.observation_space = spaces.Dict({'laser': spaces.Box(low, high), 'entropy': spaces.Box(low_coverage, high_coverage), 'coverage': spaces.Box(low_coverage, high_coverage)})
        self.observation_space = spaces.Dict({'map': spaces.Box(low=0, high=255,
                                            shape=(1, 96, 96), dtype=numpy.uint8)})


        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))

        # Rewards
        self.forwards_reward = rospy.get_param("/turtlebot3/forwards_reward")
        self.turn_reward = rospy.get_param("/turtlebot3/turn_reward")
        self.end_episode_points = rospy.get_param("/turtlebot3/end_episode_points")
        self.save_action = 0

        self.pub_closing = rospy.Publisher('/custom/closing', Bool, queue_size=20)

        self.cumulated_steps = 0.0
        self.laser_filtered_pub = rospy.Publisher('/turtlebot3/laser/scan_filtered', LaserScan, queue_size=1)
        self.update_rate_real = 5
        rospy.Subscriber("/gazebo/performance_metrics", PerformanceMetrics, self.set_rate_real_time)

        self.map_coverage = 0
        self.map = numpy.zeros((96,96,1))
        self.last_coverage = 0
        rospy.Subscriber('/map', OccupancyGrid, self.subscriber_map)


        self.covariance = []
        rospy.Subscriber('/odom', Odometry, self.subscriber_odom)

        self.actual_entropy = 0
        self.last_entropy = 1
        self.x = 0
        self.y = 0
        self.dirx = 1
        self.diry = 0
        rospy.Subscriber("/turtlebot3_slam_gmapping/entropy", Float64, self.subscriber_entropy)


    def subscriber_odom(self, data):
        self.covariance = data.pose.covariance
        self.x = data.pose.pose.position.x
        #print(f"X:{self.x},Y:{self.y}")
        self.y = data.pose.pose.position.y
        self.orx = data.pose.pose.orientation.x
        self.ory = data.pose.pose.orientation.y
        self.orw = data.pose.pose.orientation.w
        self.orz = data.pose.pose.orientation.z
        self.dirx = round(1-2*(self.ory*self.ory+self.orz*self.orz))
        self.diry = round(2*(self.orx*self.ory+self.orw*self.orz))
        #print(f"X:{self.orx},Y:{self.ory},Z:{self.orz},W:{self.orw}")
        #print(f"Direction:X{1-2*(self.ory*self.ory+self.orz*self.orz)}")
        #print(f"Direction:Y:{2*(self.orx*self.ory+self.orw*self.orz)}")
        #print(f"Direction:Z:{2*(self.orx*self.orz-self.orw*self.ory)}")
        #print(f"Direction:{self.dirx},{self.diry}")


    def subscriber_map(self, data):
        #numpy array 384x384 with -1 for unknown, 100, for occupied and 0 for free
        pc = ros_numpy.numpify(data)
        pc = pc.filled(-1)
        pc[pc == 0] = 140
        pc[pc == -1] = 0
        pc = pc.astype(numpy.uint8)
        pc[pc == 100] = 255

        #now pc has 0 for unknown, 125 for free and 255 for occupied
        pc = pc.reshape(96, 4, 96, 4).max(axis=(1, 3))
        image_x_robot = int((self.x+10)/20 * 96)
        image_y_robot = int((self.y+10)/20 *96)
        pc[image_y_robot, image_x_robot] = 70
        #This paints the direction the robot is heading to, it should not come to the case of taking modulu here but need to rework this
        #add extra dimension to map, so stable baselines recognizes it as 1 channel 0-255 greyscale image
        pc[(image_y_robot+self.diry)%96,(image_x_robot+self.dirx)%96] = 90
        image_with_channel = numpy.expand_dims(pc, axis=0)
        self.map = image_with_channel

        #plt.figure(figsize=(10, 10))  # Optional: To make the plot larger
        #plt.imshow(pc, cmap='viridis', interpolation='none')
        #plt.colorbar(label='Value')
        #plt.grid(which='both', color='grey', linestyle='-', linewidth=0.5)
        #plt.show()

        neg_count = len(list(filter(lambda x: (x < 0), data.data)))
        total = len(data.data)
        completed = 0
        if total > 0:
            completed = (100 - ( neg_count * 100 / total )) / 100

        self.map_coverage = completed


    def subscriber_entropy(self, data):
        self.actual_entropy = data.data / 5


    def set_rate_real_time(self, data):
        if (data.real_time_factor > 0):
            self.update_rate_real = data.real_time_factor * 5
        else:
            self.update_rate_real = 5


    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        self.move_base( self.init_linear_forward_speed,
                        self.init_linear_turn_speed,
                        epsilon=0.05,
                        update_rate=self.update_rate_real)

        return True


    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # For Info Purposes
        bool_closing = Bool()
        bool_closing.data = False
        self.pub_closing.publish(bool_closing)
        self.reset_gmapping.publish(String())
        rospy.sleep(1.5)

        self.cumulated_reward = 0.0
        self.map_coverage = 0
        self.last_coverage = 0
        self.save_action = 0
        self.actual_entropy = 0
        self.last_entropy = 0
        self.covariance = []

        # Set to false Done, because its calculated asyncronously
        self._episode_done = False


    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the turtlebot3
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """
        self.save_action = action
        rospy.logdebug("Start Set Action ==>"+str(action))
        # We convert the actions to speed movements to send to the parent class CubeSingleDiskEnv
        if action == 0: #FORWARD
            linear_speed = self.linear_forward_speed
            angular_speed = 0.0
            self.last_action = "FORWARDS"
        elif action == 1: #LEFT
            linear_speed = self.linear_turn_speed
            angular_speed = self.angular_speed
            self.last_action = "TURN_LEFT"
        elif action == 2: #RIGHT
            linear_speed = self.linear_turn_speed
            angular_speed = -1*self.angular_speed
            self.last_action = "TURN_RIGHT"

        # We tell TurtleBot3 the linear and angular speed to set to execute
        self.move_base(linear_speed, angular_speed, epsilon=0.05, update_rate=self.update_rate_real)
        rospy.logdebug("END Set Action ==>"+str(action))


    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        TurtleBot3Env API DOCS
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")
        # We get the laser scan data
        laser_scan = self.get_laser_scan()

        discretized_observations = self.discretize_scan_observation(    laser_scan,
                                                                        self.new_ranges
                                                                        )

        rospy.logdebug("Observations==>"+str(discretized_observations))
        rospy.logdebug("END Get Observation ==>")
        return discretized_observations


    def _is_done(self, observations):

        #for obs in observations['laser']:
        for obs in self.obsLaser:
            if obs < self.min_laser_value / self.max_laser_value:
                self._episode_done = True

        if self._episode_done:
            rospy.logerr("TurtleBot3 is Too Close to wall==>")

        return self._episode_done


    def _compute_reward(self, observations, done):
        self.cumulated_steps += 1
        d_opt = 0
        for cov in self.covariance:
            if cov > 0:
                d_opt += math.log(cov)
        #Stop divide by zero error
        if len(self.covariance) == 0:
            return 0
        d_opt = math.exp(d_opt / len(self.covariance))
        if not done:
            obs_coverage = self.map_coverage
            delta_coverage = obs_coverage - self.last_coverage
            self.last_coverage = obs_coverage
            #Some new part of map is discovered
            if delta_coverage > 0:
                #stop division by Zero
                #if (d_opt == 0):
                    #return 0
                reward = delta_coverage #math.tanh(1/d_opt)
                self.cumulated_reward += reward
                #print(f"Sum Reward episode: {self.cumulated_reward}")
                return reward
            else:
                #If nothing of map is discoverd -> no reward
                return 0
        else:
            #If robot crashed into reward no negative penalty because of driving circles
            return 0 
            #reward = -1*self.end_episode_points


        #print("reward=" + str(reward))
        self.cumulated_reward += reward
        #print("Cumulated_reward=" + str(self.cumulated_reward))
        #print("Cumulated_steps=" + str(self.cumulated_steps))

        return reward


    # Internal TaskEnv Methods

    def discretize_scan_observation(self,data,new_ranges):
        """
        Discards all the laser readings that are not multiple in index of new_ranges
        value.
        """
        self._episode_done = False
        
        discretized_ranges = []
       
        for i, item in enumerate(data.ranges):
            if (i == 0 or i == 29 or i == 45 or i == 90 or i == 270 or i == 315 or i == 329):
                if item == float ('Inf') or numpy.isinf(item):
                    discretized_ranges.append(1)
                elif numpy.isnan(item):
                    discretized_ranges.append(round(self.min_laser_value / self.max_laser_value, 1))
                else:
                    discretized_ranges.append(round(item / self.max_laser_value, 1))

                if (self.min_laser_value > item > 0):
                    self._episode_done = True
                    rospy.logerr("done Validation >>> item=" + str(item)+"< "+str(self.min_range))

                        
        self.publish_filtered_laser_scan(   laser_original_data=data,
                                            new_filtered_laser_range=discretized_ranges)
        
        new_ranges = [discretized_ranges[4], discretized_ranges[5], discretized_ranges[0], discretized_ranges[2], discretized_ranges[3]]

        #return {"laser": new_ranges, "map": self.map, "pose":[self.x,self.y,self.orx,self.ory,self.orw]}
        self.obsLaser = new_ranges
        return {"map": self.map}


    def publish_filtered_laser_scan(self, laser_original_data, new_filtered_laser_range):
        rospy.logdebug("new_filtered_laser_range==>"+str(new_filtered_laser_range))
        
        laser_filtered_object = LaserScan()

        h = Header()
        h.stamp = rospy.Time.now() # Note you need to call rospy.init_node() before this will work
        h.frame_id = laser_original_data.header.frame_id
        
        laser_filtered_object.header = h
        laser_filtered_object.angle_min = laser_original_data.angle_min
        laser_filtered_object.angle_max = laser_original_data.angle_max
        
        new_angle_incr = abs(laser_original_data.angle_max - laser_original_data.angle_min) / len(new_filtered_laser_range)
        
        #laser_filtered_object.angle_increment = laser_original_data.angle_increment
        laser_filtered_object.angle_increment = new_angle_incr
        laser_filtered_object.time_increment = laser_original_data.time_increment
        laser_filtered_object.scan_time = laser_original_data.scan_time
        laser_filtered_object.range_min = laser_original_data.range_min
        laser_filtered_object.range_max = laser_original_data.range_max
        
        laser_filtered_object.ranges = []
        laser_filtered_object.intensities = []
        for item in new_filtered_laser_range:
            if item == 0.0:
                laser_distance = 0.1
            else:
                laser_distance = item
            laser_filtered_object.ranges.append(laser_distance)
            laser_filtered_object.intensities.append(item)
        
        self.laser_filtered_pub.publish(laser_filtered_object)
