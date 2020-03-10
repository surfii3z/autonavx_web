'''
This code is HW8 for the Autonomous Navigation for Flying Robot on edx by TUM
'''

import math
import numpy as np
from plot import plot, plot_trajectory, plot_covariance_2d
        
class Pose2D:
    ''' Operation of 2D rotation matrix'''
    def __init__(self, rotation, translation):
        self.rotation = rotation
        self.translation = translation
    
    def inv(self):
        '''
            inversion of this Pose2D object
            
            :return - inverse of self
            '''
        inv_rotation = self.rotation.transpose()
        inv_translation = -np.dot(inv_rotation, self.translation)
        
        return Pose2D(inv_rotation, inv_translation)
    
    def yaw(self):
        from math import atan2
        return atan2(self.rotation[1,0], self.rotation[0,0])
    
    def __mul__(self, other):
        '''
            multiplication of two Pose2D objects, e.g.:
            a = Pose2D(...) # = self
            b = Pose2D(...) # = other
            c = a * b       # = return value
            
            :param other - Pose2D right hand side
            :return - product of self and other
            '''
        return Pose2D(np.dot(self.rotation, other.rotation), np.dot(self.rotation, other.translation) + self.translation)

class State:
    ''' Store the state of drone'''
    def __init__(self): 
        self.position = np.zeros((2,1))
        self.orientation = np.zeros((3,1))
        self.lin_velocity = np.zeros((2,1))
        self.ang_velocity = np.zeros((3,1))

class Controller:
    '''PD controller: Compute the control command'''
    def __init__(self):
        Kp_x = 2
        Kd_x = 1.3

        Kp_y = 1.5
        Kd_y = 1
        self.Kp_lin = np.array([[Kp_x, Kp_y]]).T
        self.Kd_lin = np.array([[Kd_x, Kd_y]]).T

        self.Kp_psi = 1.5
        self.Kd_psi = 1.4

    def rotation_mat_2D(self, yaw):
        '''
        create 2D rotation matrix from given angle
        '''
        s_yaw = math.sin(yaw)
        c_yaw = math.cos(yaw)
        
        return np.array([
                         [c_yaw, -s_yaw],
                         [s_yaw,  c_yaw]
                         ])
        
    def normalize_yaw(self, y):
        '''
        normalizes the given angle to the interval [-pi, +pi]
        '''
        import math
        while(y > math.pi):
            y -= 2 * math.pi
        while(y < -math.pi):
            y += 2 * math.pi
        return y
    
    def compute_control_command(self, t, dt, state, state_desired):
        
        err_lin = state_desired.position - state.position
        cur_yaw = state.orientation[2]

        # Rotate the error vector from drone to the desire point w.r.t to world frame to the error vector w.r.t to the drone frame
        # Note: the simulation world frame: North is X-axis, West is Y-axis
        # The yaw angle of the drone is measure from X-axis (North) e.g. if the drone faces North, yaw = 0, if the drone faces West, Yaw = 90
        err_lin_drone_frame = np.dot(self.rotation_mat_2D(cur_yaw + math.pi / 2).T, np.array((-err_lin[1], err_lin[0])))

        # Linear velocity control command
        u_lin = self.Kp_lin * (err_lin_drone_frame) + self.Kd_lin * (- state.lin_velocity)
        max_vel = 2.1
        u_xy = np.array([[max(min(u_lin[0], max_vel), -max_vel)], [max(min(u_lin[1], max_vel), -max_vel)]]) # saturate the velocity of the drone

        e_yaw = self.normalize_yaw(state_desired.orientation[2] - cur_yaw)
        u_yaw = self.Kp_psi * e_yaw + self.Kd_psi * (0 - state.ang_velocity[2])
        
        # Give piority to the yaw angle correctrion 
        if abs(e_yaw) > 1.2:    
            u_xy = np.zeros((2, 1))
            
        # plot the control command
        plot("u_x", u_xy[0])
        plot("u_y", u_xy[1])
        plot("u_yaw", u_yaw)
        return (u_xy, u_yaw)

class UserCode:
    def __init__(self):
        self.state = State()
        
        self.controller = Controller()
        
        # process noise
        pos_noise_std = 0.005
        yaw_noise_std = 0.005
        self.Q = np.array([
                           [pos_noise_std*pos_noise_std,0,0],
                           [0,pos_noise_std*pos_noise_std,0],
                           [0,0,yaw_noise_std*yaw_noise_std]
                           ])
                           
        # measurement noise
        z_pos_noise_std = 0.005
        z_yaw_noise_std = 0.03
        self.R = np.array([
                           [z_pos_noise_std*z_pos_noise_std,0,0],
                           [0,z_pos_noise_std*z_pos_noise_std,0],
                           [0,0,z_yaw_noise_std*z_yaw_noise_std]
                           ])

        # state vector [x, y, yaw] in world coordinates
        self.x = np.zeros((3,1))

        # 3x3 state covariance matrix                         
        self.sigma = 0.01 * np.identity(3)

        # The drone should follow these waypoints
        self.waypoints = [
            
             [2.5, 0.2],  [4.7, 0.3], [3.5, 1.9], [1, 3.5], [4.5, 3.5], # M
             
             [6, 4.5], 
             
             [7, 5.5], [4, 5.5], [4, 7.5], [4, 8.5],  [7, 8.5],  # U
             
             [8.25, 9.1],
             
             [9.5, 9.7], [9.5, 12.5], [8, 11.3], [6.2, 11.3]  # T
        ]
        self.index = 0

        # Store the next target point of the drone
        self.state_desired = State()
        self.state_desired.position = np.array([self.waypoints[self.index]]).T
    
    def update_index(self):
        '''
        update the target waypoint
        '''
        self.index = self.index + 1
    
    def normalizeYaw(self, y):
        '''
        normalizes the given angle to the interval [-pi, +pi]
        '''
        while(y > math.pi):
            y -= 2 * math.pi
        while(y < -math.pi):
            y += 2 * math.pi
        return y
    
    def rotation(self, yaw):
        '''
        create 2D rotation matrix from given angle
        '''
        s_yaw = math.sin(yaw)
        c_yaw = math.cos(yaw)
        
        return np.array([
                         [c_yaw, -s_yaw],
                         [s_yaw,  c_yaw]
                         ])
    
    def get_markers(self):
        '''
        place up to 30 markers in the world
        '''
        markers = [[0, 0],  [4.7, 0.3], [4.5, 3.5], # M
             
             [7, 5.5], [4, 8.5], # U
             
             [9.5, 10], [8.2, 11.3]  # T
        ]
        return markers
    
    
    def predictState(self, dt, x, u_linear_velocity, u_yaw_velocity):
        '''
        predicts the next state using the current state and
        the control inputs local linear velocity and yaw velocity
        '''
        x_p = np.zeros((3, 1))
        x_p[0:2] = x[0:2] + dt * np.dot(self.rotation(x[2]), u_linear_velocity)
        x_p[2]   = x[2]   + dt * u_yaw_velocity
        x_p[2]   = self.normalizeYaw(x_p[2])
        
        return x_p
    
    def calculatePredictStateJacobian(self, dt, x, u_linear_velocity, u_yaw_velocity):
        '''
        calculates the 3x3 Jacobian matrix for the predictState(...) function
        '''
        s_yaw = math.sin(x[2])
        c_yaw = math.cos(x[2])
        
        dRotation_dYaw = np.array([
                                   [-s_yaw, -c_yaw],
                                   [ c_yaw, -s_yaw]
                                   ])
        F = np.identity(3)
        F[0:2, 2] = dt * np.dot(dRotation_dYaw, u_linear_velocity)
                                   
        return F
    
    def predictCovariance(self, sigma, F, Q):
        '''
        predicts the next state covariance given the current covariance,
        the Jacobian of the predictState(...) function F and the process noise Q
        '''
        return np.dot(F, np.dot(sigma, F.T)) + Q
    
    def calculateKalmanGain(self, sigma_p, H, R):
        '''
        calculates the Kalman gain
        '''
        return np.dot(np.dot(sigma_p, H.T), np.linalg.inv(np.dot(H, np.dot(sigma_p, H.T)) + R))
    
    def correctState(self, K, x_predicted, z, z_predicted):
        '''
        corrects the current state prediction using Kalman gain, the measurement and the predicted measurement
            
        :param K - Kalman gain
        :param x_predicted - predicted state 3x1 vector
        :param z - measurement 3x1 vector
        :param z_predicted - predicted measurement 3x1 vector
        :return corrected state as 3x1 vector
        '''
        residual = (z - z_predicted)
        residual[2] = self.normalizeYaw(residual[2])
        
        return x_predicted + np.dot(K, residual)
    
    def correctCovariance(self, sigma_p, K, H):
        '''
        corrects the sate covariance matrix using Kalman gain and the Jacobian matrix of the predictMeasurement(...) function
        '''
        return np.dot(np.identity(3) - np.dot(K, H), sigma_p)
    
    def predictMeasurement(self, x, marker_position_world, marker_yaw_world):
        '''
        predicts a marker measurement given the current state and the marker position and orientation in world coordinates
        '''
        z_predicted = Pose2D(self.rotation(x[2]), x[0:2]).inv() * Pose2D(self.rotation(marker_yaw_world), marker_position_world) 
        
        return np.array([[z_predicted.translation[0], z_predicted.translation[1], z_predicted.yaw()]]).T
        
    def visualizeState(self):
        # visualize position state
        plot_trajectory("kalman", self.x[0:2])
        plot_covariance_2d("kalman", self.sigma[0:2,0:2])
        
    
    def calculatePredictMeasurementJacobian(self, x, marker_position_world, marker_yaw_world):
        '''
        calculates the 3x3 Jacobian matrix of the predictMeasurement(...) function using the current state and
        the marker position and orientation in world coordinates
        
        :param x - current state 3x1 vector
        :param marker_position_world - x and y position of the marker in world coordinates 2x1 vector
        :param marker_yaw_world - orientation of the marker in world coordinates
        :return - 3x3 Jacobian matrix of the predictMeasurement(...) function
        '''
        s_yaw = math.sin(x[2])
        c_yaw = math.cos(x[2])
        
        dx = marker_position_world[0] - x[0] 
        dy = marker_position_world[1] - x[1] 
        
        return np.array([
                         [-c_yaw, -s_yaw, s_yaw * dx - c_yaw * dy],
                         [ s_yaw, -c_yaw, c_yaw * dx + s_yaw * dy],
                         [     0,      0,                      -1]
                         ])
    
    def state_callback(self, t, dt, linear_velocity, yaw_velocity):
        '''
        called when a new odometry measurement arrives approx. 200Hz
    
        :param t - simulation time
        :param dt - time difference this last invocation
        :param linear_velocity - x and y velocity in local quadrotor coordinate frame (independet of roll and pitch)
        :param yaw_velocity - velocity around quadrotor z axis (independet of roll and pitch)
        '''
        F = self.calculatePredictStateJacobian(dt, self.x, linear_velocity, yaw_velocity)
        self.x = self.predictState(dt, self.x, linear_velocity, yaw_velocity)
        self.sigma = self.predictCovariance(self.sigma, F, self.Q) 
        self.visualizeState()
        
        self.kalman_state = State() 
        self.kalman_state.position = self.x[0:2] 
        self.kalman_state.orientation[2] = self.x[2] 

        # Update the waypoint if the drone flies close enough to the current target
        distance_error = self.state_desired.position - self.kalman_state.position
        if math.sqrt(distance_error[0] ** 2 + distance_error[1] ** 2) < 0.5 and self.index < len(self.waypoints) - 1:
            self.update_index()
            print("now self.index = ", self.index , "target is", self.waypoints[self.index])
        
        self.state_desired.position = np.array([self.waypoints[self.index]]).T
        target_yaw = math.atan2(self.state_desired.position[0] - self.x[0], -(self.state_desired.position[1] - self.x[1])) - math.pi / 2
        self.state_desired.orientation = np.array((0, 0, target_yaw)).T
        self.state_desired.lin_velocity = linear_velocity
        self.state_desired.ang_velocity  = np.array((0, 0, yaw_velocity)).T
        
        return self.controller.compute_control_command(t, dt, self.kalman_state, self.state_desired)


    def measurement_callback(self, marker_position_world, marker_yaw_world, marker_position_relative, marker_yaw_relative):
        '''
        called when a new marker measurement arrives max 30Hz, marker measurements are only available if the quadrotor is
        sufficiently close to a marker
            
        :param marker_position_world - x and y position of the marker in world coordinates 2x1 vector
        :param marker_yaw_world - orientation of the marker in world coordinates
        :param marker_position_relative - x and y position of the marker relative to the quadrotor 2x1 vector
        :param marker_yaw_relative - orientation of the marker relative to the quadrotor
        '''
        z = np.array([[marker_position_relative[0], marker_position_relative[1], marker_yaw_relative]]).T
        z_predicted = self.predictMeasurement(self.x, marker_position_world, marker_yaw_world)
        
        H = self.calculatePredictMeasurementJacobian(self.x, marker_position_world, marker_yaw_world)
        K = self.calculateKalmanGain(self.sigma, H, self.R)
        
        self.x = self.correctState(K, self.x, z, z_predicted)
        self.sigma = self.correctCovariance(self.sigma, K, H)
        