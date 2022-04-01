# Hoang Ho - 1001654608
# CSE 4310-001 Fundamentals of Computer Vision
# Assignment 3 - Detecting Motion and Kalman Filters
# Due April 1, 2022 by 11:59 PM

# references: https://dillhoffaj.utasites.cloud/posts/tracking/
#             https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html

import sys
import numpy as np

# Kalman Filter: linear dynamic model that models conditional probabilities following normal distributions
# probabilistic estimation technique - estimate state of dynamic system
#
# 2 assumptions: gaussian world and all models are linear, so model observations and state using normal distributions
#
# prediction step - takes motion commands into account to estimate in order to predict where system will be at next point in time
# correction step - takes into account sensor observation in order to correct potential mistakes and takes observation in account and compute predicted observation (what should i be) and compare 2 observations
#
# Xi - current state? predicted observation - state of the object where does it think position wise, etc
# Yi - current observation - measurement at step i
#
#
# super -: before we see new observation (every single observation before new observation)
# super +: current observation/after observation
# approximate xi with normal distrib
# control matrix/control vector - things we know about, not uncertainty so dont update state model based on that
class KalmanFilter:
    # kalman filter parameters
    # state: 2D position and velocity vector tracking object's current state - assume constant velocity
    # history: a list of object's previous positions for visualization purposes
    def __init__(self, state_model):
        # initial model
        self.state_model = state_model # predicted observation Xi
        self.cov_mtx = self.state_model @ (self.state_model.T) # corresponding covariance matrix
        self.history = [self.state_model] # history of tracked object
    
    # Prediction Task
    # estimate of where it is based on our model and what the sensor gives
    # where does our model think its going to go next?  given all our observations so far
    # P(Xk | Y0==y0,....,Yk-1=yk-1) = given all previous observation what is the model's estimate of the current state
    #
    # Prediction for step k is mean-x before we see new observation at step k = D multiplied by x(k-1)^+
    # x(k-1)^+ = the sample after we observed at step k-1
    # Assume the velocity is constant
    # t: delta t is a hyperparameter chosen by user
    def predict(self):
        t = 1
        
        # create the D matrix
        D = np.array([[1, 0, t, 0]
                    , [0, 1, 0, t]
                    , [0, 0, 1, 0]
                    , [0, 0, 0, 1]])

        # get the sample we observed at the previous step
        prev_step = len(self.history)-1
        prev_sample = self.history[prev_step]

        prediction = D @ prev_sample # prediction for step k

        #get uncertainty due to factors outside the system
        uncertainty = np.random.normal(0,np.array([[0.1, 0.1, 0.1, 0.1]
                                                ,  [0.1, 0.1, 0.1, 0.1]]))
        cov_prediction = D @ self.cov_mtx @ D.T + uncertainty # predict covariance mtx

        self.state_model = prediction
        self.cov_mtx = cov_prediction


    # give observation at what we're seeing at current time step and update predictions
    # covariance of measurement usually calculated by whoevers manufacturing the sensors
    # 
    # making corrections:
    # reconcile the uncertainty of predicted state with the uncertainty of new measurement = we wanted to know 
    # the distribution over the union of the two distributions (achieved by multiplying the gaussians together)
    #
    # Assumed predict() has already been called prior to taking into account new measurement
    def update(self, new_measurement):
        K_gain = self.cov_mtx @ np.linalg.pinv(self.cov_mtx + np.array([0.1, 0.1, 0.1, 0.1]
                                                                     , [0.1, 0.1, 0.1, 0.1]))

        # prediction after new measurement has been given
        new_prediction = self.state_model + K_gain @ (new_measurement - self.state_model)
        new_cov_prediction = self.cov_mtx - K_gain @ (self.cov_mtx)

        self.history.append(self.state_model)
        self.state_model = new_prediction
        self.cov_mtx = new_cov_prediction

