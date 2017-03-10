
import sys
import numpy as np

from inverse_dynamics import factory
from inverse_dynamics import utils

FACTORY = {}


class Interface(object):

    def __init__(self):
        pass

    def init_from_trajectory(self, traj_q, traj_qd, *args):
        pass

    def predict(self, q, qd, qdd):
        raise NotImplementedError

    @property
    def tau(self):
        raise NotImplementedError

    def update(self,
               q,
               qd,
               qdd_desired,
               qdd_actual,
               tau,
               tau_inverse_desired,
               tau_inverse_actual,
               tau_feedback,
               tau_feedforward):
        raise NotImplementedError


class FeedbackDoom(Interface):

    @classmethod
    def create_from_params(cls,
                           params):
        return cls(params.fdk_fb_learning_rate,
                   params.fdk_le_learning_rate,
                   params.fdk_le_alpha,
                   params.fdk_offset_forgetting_rate,
                   params.fdk_gradient_alpha,
                   params.fdk_gradient_forgetting_rate)

    def __init__(self,
                 fb_learning_rate,
                 le_learning_rate,
                 le_alpha,
                 offset_forgetting_rate,
                 gradient_alpha,
                 gradient_forgetting_rate):
        super(FeedbackDoom, self).__init__()
        self._fb_learning_rate = fb_learning_rate
        self._le_learning_rate = le_learning_rate
        self._offset = None
        self._offset_forgetting_rate = offset_forgetting_rate
        self._gradient = utils.ExponentialMovingAverage(
            gradient_alpha, gradient_forgetting_rate)
        self._tau_exp = utils.ExponentialMovingAverage(
            le_alpha, 1.0)

    def predict(self, q, qd, qdd):
        if self._offset is None:
            # If we have not optimized the feedback we return 0.
            self._offset = np.zeros_like(q)
            self._tau = np.copy(self._offset)
            return self._offset

        return self._offset

    @property
    def tau(self):
        return self._tau

    def update(self,
               q,
               qd,
               qdd_desired,
               qdd_actual,
               tau,
               tau_inverse_desired,
               tau_inverse_actual,
               tau_feedback,
               tau_feedforward):
        gradient = (qdd_actual - qdd_desired)
        self._offset *= self._offset_forgetting_rate
        self._tau_exp.update(self._offset - self._le_learning_rate * gradient)
        self._tau = self._tau_exp.data
        self._gradient.update(- self._fb_learning_rate * gradient)
        self._offset += self._gradient.data


class FeedbackPidTraj(Interface):

    @classmethod
    def create_from_params(cls,
                           params):
        return cls(params.fdk_fb_p_gain,
                   params.fdk_fb_d_gain,
                   params.fdk_fb_i_gain,
                   params.fdk_le_p_gain,
                   params.fdk_le_d_gain,
                   params.fdk_le_i_gain,
                   params.fdk_le_alpha)

    def __init__(self,
                 fb_p_gain,
                 fb_d_gain,
                 fb_i_gain,
                 le_p_gain,
                 le_d_gain,
                 le_i_gain,
                 le_alpha):
        super(FeedbackPidTraj, self).__init__()
        self._is_initialized = False
        self._fb_p_gain = fb_p_gain
        self._fb_d_gain = fb_d_gain
        self._fb_i_gain = fb_i_gain
        self._le_p_gain = le_p_gain
        self._le_d_gain = le_d_gain
        self._le_i_gain = le_i_gain
        self._pos = 0
        self._tau_exp = utils.ExponentialMovingAverage(
            le_alpha, 1.0)

    def init_from_trajectory(self, traj_q, traj_qd, *args):
        self._traj_q = traj_q
        self._traj_qd = traj_qd

    def predict(self, q, qd, qdd):
        if not self._is_initialized:
            self._fb_p_gain = np.diag([self._fb_p_gain] * q.size)
            self._fb_d_gain = np.diag([self._fb_d_gain] * q.size)
            self._fb_i_gain = np.diag([self._fb_i_gain] * q.size)
            self._le_p_gain = np.diag([self._le_p_gain] * q.size)
            self._le_d_gain = np.diag([self._le_d_gain] * q.size)
            self._le_i_gain = np.diag([self._le_i_gain] * q.size)
            self._tau_i = None
            self._tau_i = np.zeros_like(q)
            self._is_initialized = True
        q_next = self._traj_q[self._pos]
        qd_next = self._traj_qd[self._pos]

        # high gain
        tau_p = np.dot(self._le_p_gain, q_next - q)
        tau_d = np.dot(self._le_d_gain, qd_next - qd)
        self._tau_exp.update(self._tau_i + np.dot(
            self._le_i_gain, q_next - q) + tau_p + tau_d)
        self._tau = self._tau_exp.data

        tau_p = np.dot(self._fb_p_gain, q_next - q)

        tau_d = np.dot(self._fb_d_gain, qd_next - qd)
        self._tau_i += np.dot(self._fb_i_gain, q_next - q)
        tau = tau_p + tau_d + self._tau_i
        return tau

    @property
    def tau(self):
        return self._tau

    def update(self,
               q,
               qd,
               qdd_desired,
               qdd_actual,
               tau,
               tau_inverse_desired,
               tau_inverse_actual,
               tau_feedback,
               tau_feedforward):
        # There is nothing happening during the update step.
        self._pos += 1


class FeedbackPid(Interface):

    @classmethod
    def create_from_params(cls,
                           params):
        return cls(params.fdk_fb_p_gain,
                   params.fdk_fb_d_gain,
                   params.fdk_fb_i_gain,
                   params.fdk_le_p_gain,
                   params.fdk_le_d_gain,
                   params.fdk_le_i_gain,
                   params.fdk_le_alpha)

    def __init__(self,
                 fb_p_gain,
                 fb_d_gain,
                 fb_i_gain,
                 le_p_gain,
                 le_d_gain,
                 le_i_gain,
                 le_alpha):
        super(FeedbackPid, self).__init__()
        self._is_initialized = False
        self._fb_p_gain = fb_p_gain
        self._fb_d_gain = fb_d_gain
        self._fb_i_gain = fb_i_gain
        self._le_p_gain = le_p_gain
        self._le_d_gain = le_d_gain
        self._le_i_gain = le_i_gain
        self._tau_i = None
        self._dt = 1.0
        self._tau_exp = utils.ExponentialMovingAverage(
            le_alpha, 1.0)

    def _integrate(self, q, qd, qdd):
        q = q + self._dt * qd + 0.5 * (self._dt**2) * qdd
        qd = qd + self._dt * qdd
        return q, qd

    def predict(self, q, qd, qdd):
        if not self._is_initialized:
            self._fb_p_gain = np.diag([self._fb_p_gain] * q.size)
            self._fb_d_gain = np.diag([self._fb_d_gain] * q.size)
            self._fb_i_gain = np.diag([self._fb_i_gain] * q.size)
            self._le_p_gain = np.diag([self._le_p_gain] * q.size)
            self._le_d_gain = np.diag([self._le_d_gain] * q.size)
            self._le_i_gain = np.diag([self._le_i_gain] * q.size)
            self._tau_i = np.zeros_like(q)
            self._is_initialized = True

        q_next, qd_next = self._integrate(q, qd, qdd)
        # high gain
        tau_p = np.dot(self._le_p_gain, q_next - q)
        tau_d = np.dot(self._le_d_gain, qd_next - qd)
        self._tau_exp.update(self._tau_i + np.dot(
            self._le_i_gain, q_next - q) + tau_p + tau_d)
        self._tau = self._tau_exp.data

        tau_p = np.dot(self._fb_p_gain, q_next - q)
        tau_d = np.dot(self._fb_d_gain, qd_next - qd)
        self._tau_i += np.dot(self._fb_i_gain, q_next - q)
        tau = tau_p + tau_d + self._tau_i
        return tau

    @property
    def tau(self):
        return self._tau

    def update(self,
               q,
               qd,
               qdd_desired,
               qdd_actual,
               tau,
               tau_inverse_desired,
               tau_inverse_actual,
               tau_feedback,
               tau_feedforward):
        # There is nothing happening during the update step.
        pass


factory.register(FeedbackDoom, sys.modules[__name__])
factory.register(FeedbackPid, sys.modules[__name__])
factory.register(FeedbackPidTraj, sys.modules[__name__])


def create_from_params(params):
    if params.fdk_type not in FACTORY:
        raise Exception('The fdk_type {} is not available [{}]'.format(
            params.fdk_type, ','.join(FACTORY.keys())))

    return FACTORY[params.fdk_type].create_from_params(params)
