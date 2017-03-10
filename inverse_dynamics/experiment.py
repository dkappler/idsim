
import os
import argparse
import copy

import numpy as np

from inverse_dynamics import defines
from inverse_dynamics import utils
from inverse_dynamics import system
from inverse_dynamics import feedback
from inverse_dynamics import feedforward
from inverse_dynamics import simulation
from inverse_dynamics import noise_sensing
from inverse_dynamics import noise_forward
from inverse_dynamics import io_transformation


def run_actual_ref(exp_type, params, result):
    params_exp = params[exp_type]
    np.random.seed(params_exp.random_seed)
    name = exp_type + '_ref_actual'
    sim = simulation.Simulation.create_from_params(params_exp.sim)
    with utils.Timer(name):
        np.random.seed(params_exp.random_seed)
        system_ref = system.create_from_params(params_exp.sys_ref)
        result[name] = sim.simulation_actual(system_ref, system_ref)
    return result


def run_desired(exp_type, params, result):
    params_exp = params[exp_type]
    np.random.seed(params_exp.random_seed)
    name = exp_type + '_act_desired'
    sim = simulation.Simulation.create_from_params(params_exp.sim)
    with utils.Timer(name):
        np.random.seed(params_exp.random_seed)
        system_act = system.create_from_params(params_exp.sys_act)
        result[name] = sim.simulation_desired(system_act)

    return result


def run_actual(exp_type, params, result):
    params_exp = params[exp_type]
    np.random.seed(params_exp.random_seed)
    name = exp_type + '_act_actual'
    sim = simulation.Simulation.create_from_params(params_exp.sim)
    with utils.Timer(name):
        try:
            n_forward = noise_forward.create_from_params(
                params_exp.noise_forward)
        except AttributeError:
            n_forward = None

        try:
            n_sensing = noise_sensing.create_from_params(
                params_exp.noise_sensing)
        except AttributeError:
            n_sensing = None
        # The reference model is used to compute the forward
        # dynamics which is why we introduce the noise for the
        # forward dynamics there.
        system_ref = system.create_from_params(params_exp.sys_ref,
                                               noise_forward=n_forward)

        # The actual model provides the measurements which is why it
        # needs sensing noise.
        system_act = system.create_from_params(params_exp.sys_act,
                                               noise_sensing=n_sensing)
        result[name] = sim.simulation_actual(system_ref,
                                             system_act)
    return result


def run_actual_feedback(exp_type, params, result):
    params_exp = params[exp_type]
    np.random.seed(params_exp.random_seed)
    name = exp_type + '_act_actual_feedback'
    sim = simulation.Simulation.create_from_params(params_exp.sim)
    with utils.Timer(name):
        np.random.seed(params_exp.random_seed)
        try:
            n_forward = noise_forward.create_from_params(
                params_exp.noise_forward)
        except AttributeError:
            n_forward = None

        try:
            n_sensing = noise_sensing.create_from_params(
                params_exp.noise_sensing)
        except AttributeError:
            n_sensing = None

        # The reference model is used to compute the forward
        # dynamics which is why we introduce the noise for the
        # forward dynamics there.
        system_ref = system.create_from_params(params_exp.sys_ref,
                                               noise_forward=n_forward)

        # The actual model provides the measurements which is why it
        # needs sensing noise.
        system_act = system.create_from_params(params_exp.sys_act,
                                               noise_sensing=n_sensing)
        fdk = feedback.create_from_params(params_exp.feedback)
        result[name] = (
            sim.simulation_actual(system_ref,
                                  system_act,
                                  feedback=fdk))
    return result


def data_indirect(result_dict):
    dataset_input = {}
    dataset_input[defines.TRAJ_S] = np.zeros(
        (result_dict[defines.TRAJ_Q].shape[0], 1), dtype=np.float32)
    dataset_input[defines.TRAJ_Q] = result_dict[defines.TRAJ_Q]
    dataset_input[defines.TRAJ_QD] = result_dict[defines.TRAJ_QD]
    dataset_input[defines.TRAJ_QDD] = result_dict[defines.TRAJ_QDD_ACTUAL]

    dataset_output = {}
    dataset_output[defines.TRAJ_TAU] = (
        result_dict[defines.TRAJ_TAU] -
        result_dict[defines.TRAJ_TAU_INVERSE_ACTUAL])
    return dataset_input, dataset_output


def data_direct(result_dict):
    dataset_input = {}
    dataset_input[defines.TRAJ_S] = np.zeros(
        (result_dict[defines.TRAJ_Q][: -1].shape[0], 1), dtype=np.float32)
    dataset_input[defines.TRAJ_Q] = result_dict[defines.TRAJ_Q][: -1]
    dataset_input[defines.TRAJ_QD] = result_dict[defines.TRAJ_QD][: -1]
    dataset_input[defines.TRAJ_QDD] = result_dict[
        defines.TRAJ_QDD_DESIRED][: -1]

    dataset_output = {}
    dataset_output[defines.TRAJ_TAU] = (
        result_dict[defines.TRAJ_TAU_FEEDBACK][1:] +
        result_dict[defines.TRAJ_TAU_FEEDFORWARD][:-1])
    return dataset_input, dataset_output


def data_joined(result_dict):
    dataset_input, dataset_output = data_indirect(result_dict)
    dataset_input[defines.TRAJ_S] += 1.0
    dataset_input_indirect, dataset_output_indirect = data_direct(result_dict)
    for key in dataset_input.keys():
        print(key, dataset_input[key].shape)
        dataset_input[key] = np.vstack([dataset_input[key],
                                        dataset_input_indirect[key]])
        print(key, dataset_input[key].shape)
    for key in dataset_output.keys():
        print(dataset_output[key].shape)
        dataset_output[key] = np.vstack([dataset_output[key],
                                         dataset_output_indirect[key]])
        print(key, dataset_output[key].shape)
    return dataset_input, dataset_output


def run_actual_feedback_learn(exp_type, params, result, learn_name, data_fn):
    params_exp = params[exp_type]
    np.random.seed(params_exp.random_seed)
    sim = simulation.Simulation.create_from_params(params_exp.sim)

    io_trans = io_transformation.create_from_params(
        params_exp.io_transformation)
    fdf = feedforward.create_from_params(params_exp.feedforward,
                                         io_trans)

    for iteration in xrange(params_exp.learn_iterations):
        # We seed the iteration
        np.random.seed(params_exp.random_seed + iteration * 100)
        print('random seed ', params_exp.random_seed + iteration * 100)
        name = (exp_type + '_act_actual_feedback_learn_' +
                learn_name + '_' + str(iteration))
        with utils.Timer(name):
            try:
                n_forward = noise_forward.create_from_params(
                    params_exp.noise_forward)
            except AttributeError:
                n_forward = None

            try:
                n_sensing = noise_sensing.create_from_params(
                    params_exp.noise_sensing)
            except AttributeError:
                n_sensing = None

            # The reference model is used to compute the forward
            # dynamics which is why we introduce the noise for the
            # forward dynamics there.
            system_ref = system.create_from_params(params_exp.sys_ref,
                                                   noise_forward=n_forward)

            # The actual model provides the measurements which is why it
            # needs sensing noise.
            system_act = system.create_from_params(params_exp.sys_act,
                                                   noise_sensing=n_sensing)
            fdk = feedback.create_from_params(params_exp.feedback)
            result[name] = (
                sim.simulation_actual(system_ref,
                                      system_act,
                                      feedback=fdk,
                                      feedforward=fdf))
            dataset_input, dataset_output = data_fn(result[name])
            result[name][defines.DATASET_INPUT] = dataset_input
            result[name][defines.DATASET_OUTPUT] = dataset_output

            # We seed the training.
            np.random.seed(params_exp.random_seed + iteration * 100)
            fdf.train(dataset_input,
                      dataset_output,
                      params_exp.exp_fp_data + '_' + name)
            fdf.save(params_exp.exp_fp_data + '_' + name)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser('Experiments.')

    parser.add_argument(
        '--fp_config',
        default=os.path.join(os.path.dirname(__file__), 'config.yaml'),
        help='The scenario we run the experiment with.')

    parser.add_argument(
        '--exp_type',
        default='exp',
        help='The type of experiment we run.')

    parser.add_argument(
        '--overwrite',
        default=True,
        action='store_true',
        help='If we would rerun an experiment.')

    args, _ = parser.parse_known_args(argv)

    params = utils.params_load(args.fp_config)

    if not args.overwrite and os.path.exists(
            params[args.exp_type].exp_fp_data):
        return

    try:
        result = {}
        result = run_actual_ref(args.exp_type, params, result)
        result = run_desired(args.exp_type, params, result)
        result = run_actual(args.exp_type, params, result)
        result = run_actual_feedback(args.exp_type, params, result)
        result = run_actual_feedback_learn(
            args.exp_type,  copy.deepcopy(params),
            result, 'indirect', data_indirect)
        result = run_actual_feedback_learn(
            args.exp_type,  copy.deepcopy(params),
            result, 'direct', data_direct)
        result = run_actual_feedback_learn(
            args.exp_type,  copy.deepcopy(params),
            result, 'joined', data_joined)
        utils.pkl_save(params[args.exp_type].exp_fp_data, result)
    except:
        utils.ipdb_exception()


if __name__ == '__main__':
    main()
