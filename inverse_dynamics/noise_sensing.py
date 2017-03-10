
import sys
import numpy as np

from inverse_dynamics import factory

FACTORY = {}


class Interface(object):

    def __init__(self):
        pass

    def __call__(self, q, qd, q_sensing):
        raise NotImplementedError


class NoiseSensingNormal(Interface):

    @classmethod
    def create_from_params(cls, params):
        return cls(params.nsen_noise_scaling)

    def __init__(self, noise_scaling):
        super(NoiseSensingNormal, self).__init__()
        self._noise_scaling = noise_scaling

    def __call__(self, q, qd, q_sensing):
        # This implementation assumes that we have absolute encoders
        # thus there is no drift based on the old q_sensing values.

        # This can produce measurement noise which results in
        # really bad velocity and acceleration approximations.
        offset = self._noise_scaling * np.random.randn(q.size)

        # We make sure that the maximum scaling is met.
        offset[offset > self._noise_scaling] = self._noise_scaling
        offset[offset < -self._noise_scaling] = -self._noise_scaling
        # print(offset, q_sensing, offset + q_sensing)
        return offset + q


class NoiseSensingVelocity(Interface):

    @classmethod
    def create_from_params(cls, params):
        return cls(params.noise_scaling)

    def __init__(self, noise_scaling):
        super(NoiseSensingNormal, self).__init__()
        self._noise_scaling = noise_scaling

    def __call__(self, q, qd, q_sensing):
        # This implementation assumes that we have absolute encoders
        # thus there is no drift based on the old q_sensing values.
        offset = self._noise_scaling * np.random.rand(q.size) * np.sign(qd)
        # We make sure that the maximum scaling is met.
        offset[offset > self._noise_scaling] = self._noise_scaling
        offset[offset < -self._noise_scaling] = -self._noise_scaling
        return offset + q


factory.register(NoiseSensingNormal, sys.modules[__name__])
factory.register(NoiseSensingVelocity, sys.modules[__name__])


def create_from_params(params):
    if 'nsen_type' not in params:
        return None

    if params.nsen_type not in FACTORY:
        raise Exception('The nsen_type {} is not available [{}]'.format(
            params.nsen_type, ','.join(FACTORY.keys())))

    return FACTORY[params.nsen_type].create_from_params(params)
