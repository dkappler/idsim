import os
import copy
import argparse

from inverse_dynamics import defines
from inverse_dynamics import utils
from inverse_dynamics import visualization


def group_by_learn_type(exp_data):

    exp_data_keys = exp_data.keys()
    keys_prefix = []
    for key in exp_data_keys:
        if key.endswith('_0'):
            keys_prefix.append(key[:-2])

    exp_data_base = {}
    for key, value in exp_data.items():
        ignore = False
        for key_prefix in keys_prefix:
            if key.startswith(key_prefix):
                ignore = True
                break
        if not ignore:
            print(key)
            exp_data_base[key] = value

    result = []
    for key_prefix in keys_prefix:
        exp_data_ref = copy.deepcopy(exp_data_base)
        for key, value in exp_data.items():
            if key.startswith(key_prefix):
                exp_data_ref[key] = value
        result.append(exp_data_ref)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser('Visualization.')

    parser.add_argument(
        '--fp_config',
        default=os.path.join(os.path.dirname(__file__), 'config.yaml'),
        help='The scenario we run the experiment with.')

    parser.add_argument(
        '--fp_data',
        type=str,
        help='The data pkl file.')

    parser.add_argument(
        '--exp_type',
        default='exp',
        help='The type of experiment we run.')

    parser.add_argument(
        '--traj_q',
        action='store_true',
        help='If passed visualize the trajectory.'
    )

    parser.add_argument(
        '--traj_tau',
        action='store_true',
        help='If passed visualize the trajectory.'
    )

    parser.add_argument(
        '--dataset',
        action='store_true',
        help='If passed visualize the dataset.'
    )

    parser.add_argument(
        '--traj_tau_feedback',
        action='store_true',
        help='If passed visualize the trajectory.'
    )

    parser.add_argument(
        '--traj_tau_feedforward',
        action='store_true',
        help='If passed visualize the trajectory.'
    )

    parser.add_argument(
        '--traj_qdd_error',
        action='store_true',
        help='If passed visualize the trajectory.'
    )

    args, _ = parser.parse_known_args(argv)

    try:
        params = utils.params_load(args.fp_config)
        params_exp = params[args.exp_type]

        if args.fp_data:
            print('loading', args.fp_data)
            exp_data = utils.pkl_load(args.fp_data)
        else:
            print('loading', params_exp.exp_fp_data)
            exp_data = utils.pkl_load(params_exp.exp_fp_data)
        if args.traj_q:
            for exp_data_by_learn in group_by_learn_type(exp_data):
                traj = visualization.Trajectory2d(defines.TRAJ_Q)
                traj.plot(exp_data_by_learn)
                traj.show()
            traj = visualization.Trajectory2d(defines.TRAJ_Q)
            traj.plot(exp_data)
            traj.show()

        if args.traj_tau:
            for exp_data_by_learn in group_by_learn_type(exp_data):
                traj = visualization.Trajectory1d(defines.TRAJ_TAU)
                traj.plot(exp_data_by_learn)
                traj.show()
            traj = visualization.Trajectory1d(defines.TRAJ_TAU)
            traj.plot(exp_data)
            traj.show()

        if args.traj_tau_feedback:
            for exp_data_by_learn in group_by_learn_type(exp_data):
                traj = visualization.Trajectory1d(defines.TRAJ_TAU_FEEDBACK)
                traj.plot(exp_data_by_learn)
                traj.show()

        if args.traj_tau_feedforward:
            for exp_data_by_learn in group_by_learn_type(exp_data):
                traj = visualization.Trajectory1d(defines.TRAJ_TAU_FEEDFORWARD)
                traj.plot(exp_data_by_learn)
                traj.show()

        if args.traj_qdd_error:
            for exp_data_by_learn in group_by_learn_type(exp_data):
                traj = visualization.Trajectory1dError(
                    defines.TRAJ_QDD_ACTUAL, defines.TRAJ_QDD_DESIRED)
                traj.plot(exp_data_by_learn)
                traj.show()

        if args.dataset:
            for exp_data_by_learn in group_by_learn_type(exp_data):
                traj = visualization.Dataset()
                traj.plot(exp_data_by_learn)
                traj.show()

        visualization.show()
    except:
        utils.ipdb_exception()


if __name__ == '__main__':
    main()
