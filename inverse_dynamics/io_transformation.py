
import sys
import numpy as np

from inverse_dynamics import factory

FACTORY = {}


class Interface(object):

    def init(self, q, qd, qdd, tau):
        raise NotImplementedError

    def output_target(self, tau):
        raise NotImplementedError

    def output_predict(self, tau):
        raise NotImplementedError

    def input(self, q, qd, qdd):
        raise NotImplementedError


class IoTransformationIdentity(Interface):

    @classmethod
    def create_from_params(cls, params):
        return cls()

    def __init__(self):
        super(IoTransformationIdentity, self).__init__()

    def init(self, q, qd, qdd, tau):
        pass

    def output_target(self, tau):
        return tau

    def output_predict(self, tau):
        # Basically the inverse of of target.
        return tau

    def input(self, q, qd, qdd):
        return q, qd, qdd


class IoTransformationFixed(Interface):

    @classmethod
    def create_from_params(cls, params):
        return cls(params.iot_q, params.iot_qd,
                   params.iot_qdd,
                   params.iot_tau)

    def __init__(self, q, qd, qdd, tau):
        super(IoTransformationFixed, self).__init__()
        self._q = q
        self._qd = qd
        self._qdd = qdd
        self._tau = tau

    def init(self, q, qd, qdd, tau):
        pass

    def output_target(self, tau):
        return tau / self._tau

    def output_predict(self, tau):
        # Basically the inverse of of target.
        return tau * self._tau

    def input(self, q, qd, qdd):
        return q / self._q, qd / self._qd, qdd / self._qdd


class IoTransformationMax(Interface):

    @classmethod
    def create_from_params(cls, params):
        return cls()

    def __init__(self):
        super(IoTransformationMax, self).__init__()

    def init(self, q, qd, qdd, tau):
        self._q_max = np.abs(q).max(axis=0)
        self._qd_max = np.abs(qd).max(axis=0)
        self._qdd_max = np.abs(qdd).max(axis=0)
        self._tau_max = np.abs(qdd).max(axis=0)

    def output_target(self, tau):
        return tau * self._tau_max

    def output_predict(self, tau):
        # Basically the inverse of of target.
        return tau * self._tau_max

    def input(self, q, qd, qdd):
        return (q / self._q_max,
                qd / self._qd_max,
                qdd / self._qdd_max)


factory.register(IoTransformationIdentity, sys.modules[__name__])
factory.register(IoTransformationMax, sys.modules[__name__])
factory.register(IoTransformationFixed, sys.modules[__name__])


def create_from_params(params):
    if params.iot_type not in FACTORY:
        raise Exception('The iot_type {} is not available [{}]'.format(
            params.iot_type, ','.join(FACTORY.keys())))

    return FACTORY[params.iot_type].create_from_params(params)
