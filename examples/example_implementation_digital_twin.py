#!/usr/bin/env python

import rospy
import numpy as np
from gazebo_msgs.msg import ModelStates
import uav_radio
from rospy import Time

log_file = "rf_data_full.log"
rospy.init_node('drone_rf_python_script')

class rf_body:
    def __init__(self):
        """
        Reads the table of parameters
        """
        self.calculator = uav_radio.PathLossCalculator(reference_distance=1.0)

    def get_distance_loss(self, d):
        """
        Calculates the path loss of the copter based on the distance to the ground control station (origin of coordinates for simplicity).
        Parameters
        ----------
        d : float
            distance in m to the ground station
        Returns
        -------
        float
            returns the path loss based on distance
        """
        spatial_loss_fspl_868,_ = self.calculator.free_space_pl(distance=d, frequency=868000000)
        spatial_loss_alpha_beta_868,_ = self.calculator.log_distance_alpha_beta_pl(distance=d, frequency=868000000, scenario="lightly_hilly_rural", rx_height=30)
        spatial_loss_two_ray_868,_ = self.calculator.two_ray_pl(distance=d, frequency=868000000, h_t=1.5, h_r=30.0)
        spatial_loss_modified_two_ray_868,_ = self.calculator.modified_two_ray_pl(distance=d, frequency=868000000, h_t=1.5, h_r=30.0, G_l_custom=None, G_r_custom=None, R_custom=None, gamma_h_custom=None)
        spatial_loss_dual_slope_868,_ = self.calculator.dual_slope_pl(distance=d, frequency=868000000, gamma_1=0.74, gamma_2=2.29, default=30, d_b=9.0)
        spatial_loss_fspl_2_4,_ = self.calculator.free_space_pl(distance=d, frequency=2400000000)
        spatial_loss_alpha_beta_2_4,_ = self.calculator.log_distance_alpha_beta_pl(distance=d, frequency=2400000000, scenario="lightly_hilly_rural", rx_height=30)
        spatial_loss_two_ray_2_4,_ = self.calculator.two_ray_pl(distance=d, frequency=2400000000, h_t=1.5, h_r=30.0)
        spatial_loss_modified_two_ray_2_4,_ = self.calculator.modified_two_ray_pl(distance=d, frequency=2400000000, h_t=1.5, h_r=30.0, G_l_custom=None, G_r_custom=None, R_custom=None, gamma_h_custom=None)
        spatial_loss_dual_slope_2_4,_ = self.calculator.dual_slope_pl(distance=d, frequency=2400000000, gamma_1=0.74, gamma_2=2.29, default=30, d_b=9.0)
        spatial_loss_fspl_5_8,_ = self.calculator.free_space_pl(distance=d, frequency=5800000000)
        spatial_loss_alpha_beta_5_8,_ = self.calculator.log_distance_alpha_beta_pl(distance=d, frequency=5800000000, scenario="lightly_hilly_rural", rx_height=30)
        spatial_loss_two_ray_5_8,_ = self.calculator.two_ray_pl(distance=d, frequency=5800000000, h_t=1.5, h_r=30.0)
        spatial_loss_modified_two_ray_5_8,_ = self.calculator.modified_two_ray_pl(distance=d, frequency=5800000000, h_t=1.5, h_r=30.0, G_l_custom=None, G_r_custom=None, R_custom=None, gamma_h_custom=None)
        spatial_loss_dual_slope_5_8,_ = self.calculator.dual_slope_pl(distance=d, frequency=5800000000, gamma_1=0.74, gamma_2=2.29, default=30, d_b=9.0)

        return spatial_loss_fspl_868, spatial_loss_alpha_beta_868, spatial_loss_two_ray_868, spatial_loss_modified_two_ray_868, spatial_loss_dual_slope_868, spatial_loss_fspl_2_4, spatial_loss_alpha_beta_2_4, spatial_loss_two_ray_2_4, spatial_loss_modified_two_ray_2_4, spatial_loss_dual_slope_2_4, spatial_loss_fspl_5_8, spatial_loss_alpha_beta_5_8, spatial_loss_two_ray_5_8, spatial_loss_modified_two_ray_5_8, spatial_loss_dual_slope_5_8  

    def calc_losses(self, position):
        """
        Calculates the RF losses based on the distance to the ground station
        and the quad attitude.
        Parameters
        ----------
        position : np array [3]
            the position of the copter
        Returns
        -------
        np array [1], np array [1]
            returns the computed path losses
        """
        
        distance = np.linalg.norm(position) # we calculate the absolute distance between the origin of coordinates and the quadcopter.
        spatial_loss_fspl_868, spatial_loss_alpha_beta_868, spatial_loss_two_ray_868, spatial_loss_modified_two_ray_868, spatial_loss_dual_slope_868, spatial_loss_fspl_2_4, spatial_loss_alpha_beta_2_4, spatial_loss_two_ray_2_4, spatial_loss_modified_two_ray_2_4, spatial_loss_dual_slope_2_4, spatial_loss_fspl_5_8, spatial_loss_alpha_beta_5_8, spatial_loss_two_ray_5_8, spatial_loss_modified_two_ray_5_8, spatial_loss_dual_slope_5_8 = self.get_distance_loss(d = distance)

        return spatial_loss_fspl_868, spatial_loss_alpha_beta_868, spatial_loss_two_ray_868, spatial_loss_modified_two_ray_868, spatial_loss_dual_slope_868, spatial_loss_fspl_2_4, spatial_loss_alpha_beta_2_4, spatial_loss_two_ray_2_4, spatial_loss_modified_two_ray_2_4, spatial_loss_dual_slope_2_4, spatial_loss_fspl_5_8, spatial_loss_alpha_beta_5_8, spatial_loss_two_ray_5_8, spatial_loss_modified_two_ray_5_8, spatial_loss_dual_slope_5_8, np.round(distance,3)

class rf():
    """
        Sets the RF model ready for experimentation with the copter.
    """     
    def __init__(self):
        self.frequency = 50.0
        self.rate = rospy.Rate(self.frequency)
        self.sub = rospy.Subscriber("/gazebo/model_states", ModelStates, self.location_callback)
        self.quad_position = np.zeros(3) + 0.01 # slight offset to avoid error on path loss initial calculation
        self.quad = rf_body()
        self.name = 'drone1' # replace with the name of your entity in Gazebo/ROS

    def location_callback(self, msg):
        """
        Gets the location of the copter with ROSPY package
        Parameters
        ----------
        msg : gazebo msg
            the gazebo message to be read
        """
        ind = msg.name.index(self.name)
        positionObj = msg.pose[ind].position
        self.quad_position = np.array([positionObj.x, positionObj.y, positionObj.z])

    def rf_signal(self):
        """
        Main point of the algorithm, runs the hexacopter pose, calculates the rf gain,
        and sends it to a node or prints it.
        """

        spatial_loss_fspl_868, spatial_loss_alpha_beta_868, spatial_loss_two_ray_868, spatial_loss_modified_two_ray_868, spatial_loss_dual_slope_868, spatial_loss_fspl_2_4, spatial_loss_alpha_beta_2_4, spatial_loss_two_ray_2_4, spatial_loss_modified_two_ray_2_4, spatial_loss_dual_slope_2_4, spatial_loss_fspl_5_8, spatial_loss_alpha_beta_5_8, spatial_loss_two_ray_5_8, spatial_loss_modified_two_ray_5_8, spatial_loss_dual_slope_5_8, distance = self.quad.calc_losses(position = self.quad_position)

        current_time = Time.now().to_sec()

        # Write data to the log file
        with open(log_file, "a") as log:
            log.write(f"{current_time},{distance},{spatial_loss_fspl_868}, {spatial_loss_alpha_beta_868}, {spatial_loss_two_ray_868}, {spatial_loss_modified_two_ray_868}, {spatial_loss_dual_slope_868}, {spatial_loss_fspl_2_4}, {spatial_loss_alpha_beta_2_4}, {spatial_loss_two_ray_2_4}, {spatial_loss_modified_two_ray_2_4}, {spatial_loss_dual_slope_2_4}, {spatial_loss_fspl_5_8}, {spatial_loss_alpha_beta_5_8}, {spatial_loss_two_ray_5_8}, {spatial_loss_modified_two_ray_5_8}, {spatial_loss_dual_slope_5_8}\n")

        # Print the data
        print("--------------------------------")
        print(f"Time: {current_time:.2f} s")
        print(f"Distance: {distance:.2f} m")
        print(f"Path Loss: {spatial_loss_fspl_868:.2f} dB")

        self.rate.sleep()

if __name__ == "__main__":
    radio = rf()
    while True:
        radio.rf_signal()
        