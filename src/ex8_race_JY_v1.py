import math
import numpy as np
from plot import plot, plot_trajectory, plot_covariance_2d
        
class UserCode:
    def __init__(self):
        self.state = State()
        self.wp = [[2.5, 0.2],  [4.7, 0.3], [3.5, 1.9], [1, 3.5], [4.5, 3.5], # M
             
             [7, 5.5], [4, 5.5], [4, 8.5],  [7, 8.5],  # U
             
             [9.5, 10], [9.5, 12.5], [8.2, 11.3], [6.2, 11.3]  # T
        ]
        self.index = 0
    
    def get_markers(self):
        '''
        place up to 30 markers in the world
        '''
        markers = [
             [0, 0], [1.5, 0], [3, 0], [4.5, 0], [4, 1], [3.3, 1.7], [2, 2.5], [1, 3.5], [3, 3.5], [4.5, 3.5], # M
             
             [6, 4.5], [7, 5.5], [4, 5.5], [4, 7], [4, 8.5], [5.5, 8.5], [7, 8.5],  # U
             
             [8.5, 9], [9.5, 10], [9.5, 11.3], [9.5, 12.5], [8.5, 11.3], [7.2, 11.3], [6.2, 11.3]  # T
        ]
        
        #TODO: Add your markers where needed
       
        return markers
    
        
    def state_callback(self, t, dt, linear_velocity, yaw_velocity):
        '''
        called when a new odometry measurement arrives approx. 200Hz
    
        :param t - simulation time
        :param dt - time difference this last invocation
        :param linear_velocity - x and y velocity in local quadrotor coordinate frame (independet of roll and pitch)
        :param yaw_velocity - velocity around quadrotor z axis (independet of roll and pitch)

        :return tuple containing linear x and y velocity control commands in local quadrotor coordinate frame (independet of roll and pitch), and yaw velocity
        '''
        self.state.x = self.state.predictState(dt, self.state.x, linear_velocity, yaw_velocity)
        F = self.state.calculatePredictStateJacobian(dt, self.state.x, linear_velocity, yaw_velocity)
        self.state.sigma = self.state.predictCovariance(self.state.sigma, F, self.state.Q);
        self.state.visualizeState()
        
        ''' Control Command '''
        target_x_w = self.wp[self.index][0]
        target_y_w = self.wp[self.index][1]
        
        target_yaw = math.atan2(target_x_w - self.state.x[0], -(target_y_w - self.state.x[1])) - math.pi / 2
        eyaw = self.state.normalizeYaw(target_yaw - self.state.x[2])
        
        Kp_yaw = 5
        Kd_yaw = 4.3
        uyaw = Kp_yaw * (eyaw) + Kd_yaw * (0 - yaw_velocity)
        
        
        Kp_x = 2
        Kd_x = 1.3
        
        Kp_y = 1.5
        Kd_y = 1
        
        # Kp_y = 0.3
        # Kd_y = 0.15
        
        ex_w = target_x_w - self.state.x[0]
        ey_w = target_y_w - self.state.x[1]
        # temp = np.dot(self.state.rotation(self.state.x[2])  ,np.array((ex_w, ey_w)))
        temp = np.dot(self.state.rotation(self.state.x[2] + math.pi/2).T  , np.array((-ey_w, ex_w)))
        
        ux = Kp_x * (temp[0]) + Kd_x * (0 - linear_velocity[0])
        uy = Kp_y * (temp[1]) + Kd_y * (0 - linear_velocity[1])
        
        if abs(eyaw) > 1.2:
            uy = 0
            ux = 0
            
        
        
        
        if math.sqrt(ex_w**2 + ey_w **2) < 0.5 and self.index < len(self.wp) - 1:
            self.index = self.index + 1
            print("now self.index = ", self.index , "target is", self.wp[self.index])
    
        
        from plot import plot
        plot("ex", ex_w)
        plot("ey", ey_w)
        plot("eyaw", eyaw)
        
        plot("ux", ux)
        plot("uy", uy)
        plot("uyaw", uyaw)

        
        return np.array((ux, uy)), uyaw


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
        z_predicted = self.state.predictMeasurement(self.state.x, marker_position_world, marker_yaw_world)
        H = self.state.calculatePredictMeasurementJacobian(self.state.x, marker_position_world, marker_yaw_world)
        K = self.state.calculateKalmanGain(self.state.sigma, H, self.state.R)
        
        self.state.x = self.state.correctState(K, self.state.x, z, z_predicted)
        self.state.sigma = self.state.correctCovariance(self.state.sigma, K, H)
        
        self.state.visualizeState()

class State:
    def __init__(self):        
        #TODO: Play with the noise matrices
        
        #process noise
        pos_noise_std = 0.005
        yaw_noise_std = 0.005
        self.Q = np.array([
            [pos_noise_std*pos_noise_std,0,0],
            [0,pos_noise_std*pos_noise_std,0],
            [0,0,yaw_noise_std*yaw_noise_std]
        ]) 
        
        #measurement noise
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
    
    def normalizeYaw(self, y):
        '''
        normalizes the given angle to the interval [-pi, +pi]
        '''
        while(y > math.pi):
            y -= 2 * math.pi
        while(y < -math.pi):
            y += 2 * math.pi
        return y
    
    def visualizeState(self):
        # visualize position state
        plot_trajectory("kalman", self.x[0:2])
        plot_covariance_2d("kalman", self.sigma[0:2,0:2])
    
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
        z_predicted = Pose2D(self.rotation(x[2]), x[0:2]).inv() * Pose2D(self.rotation(marker_yaw_world), marker_position_world);
        
        return np.array([[z_predicted.translation[0], z_predicted.translation[1], z_predicted.yaw()]]).T
    
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
        
        dx = marker_position_world[0] - x[0];
        dy = marker_position_world[1] - x[1];
        
        return np.array([
            [-c_yaw, -s_yaw, -s_yaw * dx + c_yaw * dy],
            [ s_yaw, -c_yaw, -c_yaw * dx - s_yaw * dy],
            [     0,      0,                      -1]
        ])
        
class Pose2D:
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