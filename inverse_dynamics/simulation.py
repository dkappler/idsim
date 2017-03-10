
import numpy as np

from inverse_dynamics import defines


class Simulation(object):

    @classmethod
    def create_from_params(cls, params):
        return cls(params.sim_num_iteration,
                   params.sim_max_tau,
                   params.sim_max_tau_feedback,
                   params.sim_max_tau_feedforward)

    def __init__(self,
                 num_iteration,
                 max_tau,
                 max_tau_feedback,
                 max_tau_feedforward):
        self._num_iteration = num_iteration
        self._max_tau = max_tau
        self._max_tau_feedback = max_tau_feedback
        self._max_tau_feedforward = max_tau_feedforward

    def simulation_desired(self, system):
        result = {}
        traj_q, traj_qd, traj_qdd = system.desired_trajectory(
            system.q, system.qd, self._num_iteration)
        result[defines.TRAJ_Q] = traj_q
        result[defines.TRAJ_QD] = traj_qd
        result[defines.TRAJ_QDD_DESIRED] = traj_qdd
        return result

    def _bound(self, tau):
        tau_bound = np.copy(tau)
        tau_bound[tau > self._max_tau_feedback] = self._max_tau_feedback
        tau_bound[tau < -self._max_tau_feedback] = -self._max_tau_feedback
        return tau_bound

    def simulation_actual(self,
                          system_ref,
                          system_act,
                          feedback=None,
                          feedforward=None):
        result = {}
        traj_q = np.zeros(((self._num_iteration,) + system_ref.q.shape),
                          dtype=system_ref.q.dtype)
        traj_qd = np.zeros(((self._num_iteration,) + system_ref.q.shape),
                           dtype=system_ref.q.dtype)

        traj_q_sensing = np.zeros(
            ((self._num_iteration,) + system_act.q.shape),
            dtype=system_act.q.dtype)
        traj_qd_sensing = np.zeros(
            ((self._num_iteration,) + system_act.q.shape),
            dtype=system_act.q.dtype)
        traj_qdd_desired = np.zeros(
            ((self._num_iteration,) + system_act.q.shape),
            dtype=system_act.q.dtype)
        traj_qdd_actual = np.zeros(
            ((self._num_iteration,) + system_act.q.shape),
            dtype=system_act.q.dtype)

        traj_tau = np.zeros(
            ((self._num_iteration,) + system_act.q.shape),
            dtype=system_act.q.dtype)
        traj_tau_inverse_desired = np.zeros(
            ((self._num_iteration,) + system_act.q.shape),
            dtype=system_act.q.dtype)
        traj_tau_inverse_actual = np.zeros(
            ((self._num_iteration,) + system_act.q.shape),
            dtype=system_act.q.dtype)
        traj_tau_feedback = np.zeros(
            ((self._num_iteration,) + system_act.q.shape),
            dtype=system_act.q.dtype)
        traj_tau_feedforward = np.zeros(
            ((self._num_iteration,) + system_act.q.shape),
            dtype=system_act.q.dtype)

        if feedback is not None:
            feedback.init_from_trajectory(
                *system_act.desired_trajectory(
                    system_act.q, system_act.qd, self._num_iteration))

        for pos in xrange(self._num_iteration):
            q_sensing = system_act.q_sensing
            qd_sensing = system_act.qd_sensing
            q = system_act.q
            qd = system_act.qd

            qdd_desired = system_act.policy_evaluation(q_sensing,
                                                       qd_sensing)

            tau_inverse_desired = system_act.dynamics_inverse(
                q_sensing, qd_sensing, qdd_desired)
            tau_inverse_desired = self._bound(tau_inverse_desired)

            tau = np.zeros_like(tau_inverse_desired)
            tau += tau_inverse_desired

            tau_feedback = np.zeros_like(tau)
            if feedback is not None:
                tau_feedback = self._bound(feedback.predict(
                    q_sensing, qd_sensing, qdd_desired))
            tau += tau_feedback

            tau_feedforward = np.zeros_like(tau)
            if feedforward is not None:
                tau_feedforward = self._bound(feedforward.predict(
                    q_sensing, qd_sensing, qdd_desired))
            tau += tau_feedforward

            traj_q[pos] = q
            traj_qd[pos] = qd

            traj_q_sensing[pos] = q_sensing
            traj_qd_sensing[pos] = qd_sensing
            traj_qdd_desired[pos] = qdd_desired

            traj_tau[pos] = tau
            traj_tau_feedback[pos] = tau_feedback
            if feedback is not None:
                traj_tau_feedback[pos] = self._bound(feedback.tau)

            traj_tau_inverse_desired[pos] = tau_inverse_desired
            traj_tau_feedforward[pos] = tau_feedforward

            # Now we compute the system update and we are essentially getting
            # results for pos + 1.
            qdd = system_ref.dynamics_forward(q, qd, tau)
            q, qd = system_act.integrate(q, qd, qdd)

            system_act.update_state(q, qd)
            # The actual acceleration is the outcome of the previous actual
            # acceleration which is why we store it with that stamp.
            traj_qdd_actual[pos] = system_act.qdd_sensing
            tau_inverse_actual = self._bound(system_act.dynamics_inverse(
                q_sensing, qd_sensing, system_act.qdd_sensing))
            traj_tau_inverse_actual[pos] = tau_inverse_actual

            if feedback is not None:
                feedback.update(q_sensing,
                                qd_sensing,
                                qdd_desired,
                                system_act.qdd_sensing,
                                tau,
                                tau_inverse_desired,
                                tau_inverse_actual,
                                tau_feedback,
                                tau_feedforward)

        result[defines.TRAJ_Q] = traj_q
        result[defines.TRAJ_QD] = traj_qd
        result[defines.TRAJ_Q_SENSING] = traj_q_sensing
        result[defines.TRAJ_QD_SENSING] = traj_qd_sensing
        result[defines.TRAJ_QDD_DESIRED] = traj_qdd_desired
        result[defines.TRAJ_QDD_ACTUAL] = traj_qdd_actual

        result[defines.TRAJ_TAU] = traj_tau
        result[defines.TRAJ_TAU_INVERSE_ACTUAL] = traj_tau_inverse_actual
        result[defines.TRAJ_TAU_INVERSE_DESIRED] = traj_tau_inverse_desired
        result[defines.TRAJ_TAU_FEEDBACK] = traj_tau_feedback
        result[defines.TRAJ_TAU_FEEDFORWARD] = traj_tau_feedforward

        return result
