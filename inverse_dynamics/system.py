
import sys
import numpy as np

from inverse_dynamics import factory

FACTORY = {}


class Interface(object):

    def __init__(self, dt, q_start, qd_start, noise_sensing=None,
                 noise_forward=None):
        self._dt = np.array(dt)
        self._noise_sensing = noise_sensing
        self._noise_forward = noise_forward

        # We initialize both states.
        self._q = np.array(q_start)
        self._qd = np.array(qd_start)
        self._q_sensing = np.array(q_start)
        self._qd_sensing = np.array(qd_start)
        self._qdd_sensing = np.zeros_like(q_start)

    def dynamics_inverse(self, q, qd, qdd):
        raise NotImplementedError

    def dynamics_forward(self, q, qd, qdd):
        raise NotImplementedError

    def policy_evaluation(self, q, qd):
        raise NotImplementedError

    def integrate(self, q, qd, qdd):
        q = q + self._dt * qd + 0.5 * (self._dt**2) * qdd
        qd = qd + self._dt * qdd
        return q, qd

    def update_state(self, q, qd):
        q_sensing = np.copy(q)

        if self._noise_sensing is not None:
            # We can only sense positions.
            q_sensing = self._noise_sensing(q, qd, self._q_sensing)

        qd_sensing = (q_sensing - self._q_sensing) / self._dt
        self._qdd_sensing = (qd_sensing - self._qd_sensing) / self._dt
        self._qd_sensing = qd_sensing
        self._q_sensing = q_sensing

        self._q = q
        self._qd = np.copy(qd)

    def desired_trajectory(self, q, qd, num_iteration):
        traj_q = np.zeros(((num_iteration,) + q.shape), dtype=q.dtype)
        traj_qd = np.zeros(((num_iteration,) + q.shape), dtype=q.dtype)
        traj_qdd = np.zeros(((num_iteration,) + q.shape), dtype=q.dtype)
        for pos in xrange(num_iteration):
            qdd = self.policy_evaluation(q, qd)
            traj_q[pos] = q
            traj_qd[pos] = qd
            traj_qdd[pos] = qdd
            q, qd = self.integrate(q, qd, qdd)
        return traj_q, traj_qd, traj_qdd

    @property
    def q(self):
        return self._q

    @property
    def qd(self):
        return self._qd

    @property
    def q_sensing(self):
        return self._q_sensing

    @property
    def qd_sensing(self):
        return self._qd_sensing

    @property
    def qdd_sensing(self):
        return self._qdd_sensing


class SystemPd(Interface):

    @classmethod
    def create_from_params(cls,
                           params,
                           noise_sensing=None,
                           noise_forward=None):
        return cls(params.sys_dt,
                   params.sys_q_start,
                   params.sys_qd_start,
                   params.sys_p_gain,
                   params.sys_d_gain,
                   params.sys_q_target,
                   params.sys_qd_target,
                   params.sys_mass,
                   noise_sensing,
                   noise_forward)

    def __init__(self, dt, q_start, qd_start, p_gain, d_gain,
                 q_target, qd_target, mass,
                 noise_sensing=None,
                 noise_forward=None):
        super(SystemPd, self).__init__(dt,
                                       q_start,
                                       qd_start,
                                       noise_sensing,
                                       noise_forward)
        self._p_gain = np.array(p_gain)
        self._d_gain = np.array(d_gain)
        self._q_target = np.array(q_target)
        self._qd_target = np.array(qd_target)
        self._mass = np.diag(mass)

    def policy_evaluation(self, q, qd):
        return (self._p_gain * (self._q_target - q) +
                self._d_gain * (self._qd_target - qd))

    def _dynamics_forward(self, q, qd, tau):
        return np.dot(np.linalg.inv(self._mass), tau)

    def dynamics_forward(self, q, qd, tau):
        qdd = self._dynamics_forward(q, qd, tau)
        if self._noise_forward is not None:
            return self._noise_forward(q, qd, qdd, tau, self._dynamics_forward)
        return qdd

    def dynamics_inverse(self, q, qd, qdd):
        return np.dot(self._mass, qdd)


factory.register(SystemPd, sys.modules[__name__])


def create_from_params(params, *args, **kwargs):
    if params.sys_type not in FACTORY:
        raise Exception('The sys_type {} is not available [{}]'.format(
            params.sys_type, ','.join(FACTORY.keys())))

    return FACTORY[params.sys_type].create_from_params(params, *args, **kwargs)
