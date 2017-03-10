
import sys

import numpy as np

from inverse_dynamics import factory

FACTORY = {}


class Interface(object):

    def __init__(self):
        pass

    def __call__(self, q, qd, qdd, tau, dynamics_forward_fn):
        raise NotImplementedError


class NoiseForwardBasic(Interface):

    @classmethod
    def create_from_params(cls, params):
        return cls(params.nfor_noise_max,
                   params.nfor_friction,
                   params.nfor_stiction)

    def __init__(self, noise_max, friction, stiction):
        super(NoiseForwardBasic, self).__init__()
        self._noise_max = noise_max
        self._friction = friction
        self._stiction = stiction

    def __call__(self, q, qd, qdd, tau, dynamics_forward_fn):
        if self._friction > 0:
            tmp_friction = np.zeros_like(qdd)

            # state dependent friction
            if q[0] < 0.2:
                tmp_friction[0] = (
                    10 * np.exp(self._friction) * np.sin(q[0]) ** 2)

            if q[1] < 0.2:
                tmp_friction[1] = 10 * self._friction * np.cos(q[1]) ** 2

            if q[1] < 0.4 and q[1] > 0.2:
                # negative friction something attracting :)
                tmp_friction[1] = (
                    -1.0 * self._friction ** 2 * np.sin(q[1]) ** 2)

            if q[1] < 1.2 and q[1] > 0.4:
                # in this state the friction on state 1 depends on state 2
                tmp_friction[0] = (
                    1.0 * max(0.0, self._friction) * np.sin(q[1]) ** 2)

            qdd = dynamics_forward_fn(q, qd, tau - tmp_friction * qd)

        if self._noise_max > 0:
            noise = np.random.randn(qdd.size) * self._noise_max
            noise[noise > self._noise_max] = self._noise_max
            noise[noise < -self._noise_max] = -self._noise_max
            qdd += noise

        if self._stiction > 0:
            # we need a lot of torque to get moving
            if np.abs(qd[0]) < 0.0001:
                if q[0] < 0.2:
                    if tau[0] < 100 * self._stiction:
                        qdd[0] = 0

            # we need a lot of torque to get moving
            if np.abs(qd[1]) < 0.0001:
                if q[1] < 0.2:
                    if tau[1] < 50 * self._stiction:
                        qdd[1] = 0
        return qdd


factory.register(NoiseForwardBasic, sys.modules[__name__])


def create_from_params(params):
    if 'nfor_type' not in params:
        return None

    if params.nfor_type not in FACTORY:
        raise Exception('The nfor_type {} is not available [{}]'.format(
            params.nfor_type, ','.join(FACTORY.keys())))

    return FACTORY[params.nfor_type].create_from_params(params)
