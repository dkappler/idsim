
import yaml
import os
import argparse
import shutil
import easydict

import numpy as np

import matplotlib
matplotlib.use('Agg')

font = {'family': 'normal',
        'weight': 'normal',
        'size': 16}

matplotlib.rc('font', **font)

import matplotlib.pyplot as plt  # noqa


from inverse_dynamics import defines
from inverse_dynamics import utils
from inverse_dynamics import visualization


def collect_runs_root(dp_exp_root):
    # Collect all experiments for processing on a single machine.
    if not os.path.exists(dp_exp_root):
        raise Exception('The dp_exp_root {} does not exist.'.format(
            dp_exp_root))

    runs = {}
    for file_name in sorted(os.listdir(dp_exp_root)):
        file_path = os.path.join(dp_exp_root, file_name)
        if not os.path.isdir(file_path):
            continue
        res = file_prefix_split(file_name)
        if res is None:
            continue
        file_path = get_data_file(file_path)
        if file_path is None:
            continue
        if res.file_prefix in runs:
            runs[res.file_prefix][res.run_index] = file_path
        else:
            runs[res.file_prefix] = {res.run_index: file_path}
    return runs


def collect_runs_single(fp_config, exp_type):
    # Collect a sinlge run to perform evaluation on the cluster.
    params = utils.params_load(fp_config)
    params_exp = params[exp_type]

    if (not os.path.exists(params_exp.exp_fp_data) and
            not os.path.exists(params_exp.exp_fp_data + '.gz')):
        raise Exception('The exp_fp_data {} does not exist.'.format(
            params_exp.exp_fp_data))
    dp_data = os.path.dirname(params_exp.exp_fp_data)
    for dp_name in os.listdir(dp_data):
        dir_path = os.path.join(dp_data, dp_name)
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)

    file_name = os.path.basename(os.path.dirname(params_exp.exp_fp_data))
    res = file_prefix_split(file_name)
    if res is None:
        return {}

    runs = {
        res.file_prefix: {res.run_index: params_exp.exp_fp_data}
    }

    return runs


def file_prefix_split(file_name):
    # We run every experiment with n different random seeds in order to
    # get an estimate of variance.
    # The random seed is split with _.
    if '_' not in file_name:
        return None
    file_prefix = '_'.join(file_name.split('_')[:-1])
    run = file_name.split('_')[-1]
    res = easydict.EasyDict()
    res.file_prefix = file_prefix
    res.run_index = int(run)
    return res


def get_data_file(file_path):
    # We assume that there is exactly one pkl file in every folder.
    result = None
    for file_name in sorted(os.listdir(file_path)):
        if file_name.endswith('pkl'):
            if result is not None:
                raise Exception('We have too many pkl files in the folder.')
            result = os.path.join(file_path, file_name)

        if file_name.endswith('pkl.gz'):
            if result is not None:
                raise Exception('We have too many pkl.gz files in the folder.')
            result = os.path.join(file_path, file_name)

    if result is None:
        print('No data file found for {}.'.format(file_path))
        return result
    return result


def get_learning_data(data_type, data, substring_exclude=None):
    result = {}
    for key, value in sorted(data.items()):

        if substring_exclude is not None and substring_exclude in key:
            continue
        if data_type not in key:
            continue

        # special case for trajectory plotting
        if substring_exclude is not None:
            if not key.endswith(data_type):
                continue

        try:
            run = int(key.split('_')[-1])
        except ValueError:
            # special case for trajectory plotting
            run = 0
        if run in result:
            raise Exception('We have some run twice {}.'.format(run))
        result[run] = value
    if np.max(result.keys()) != len(result) - 1:
        raise Exception('We are missing a run {} vs {}.'.format(
            np.max(result.keys()), len(result) - 1))
    return [value for _, value in sorted(result.items())]


def get_ref_data(data):
    for key, value in data.items():
        if key.endswith('_ref_actual'):
            return value
    raise Exception('No reference data found.')


def get_file_path(dir_output, file_prefix, data_type, ext, run):
    file_prefix = os.path.basename(file_prefix)
    fp_output = os.path.join(dir_output,
                             file_prefix + data_type + ext + '_' + run +
                             '.pkl')
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)
    return fp_output


def get_file_path_cache(dir_output, file_name):
    file_path = os.path.join(dir_output, file_name)
    if os.path.exists(file_path):
        return True, file_path
    if os.path.exists(file_path + '.gz'):
        return True, file_path
    return False, file_path


def _compute_mean(dir_output, data_type, data_key, runs, callback_fn,
                  substring_exclude=None):
    results = easydict.EasyDict()
    results.data_type = data_type
    results.data = {}
    results.mean = {}
    results.std = {}
    results.params = {}
    for file_prefix, run_dict in runs.items():
        results_per_run = []
        params = None
        results_per_run = []
        for run, file_path in run_dict.items():

            if params is None:
                params = utils.yaml_load(
                    os.path.join(
                        os.path.dirname(file_path),
                        defines.EXPERIMENT_TEMPLATE_PARAMS_YAML))

            file_data_cache = get_file_path(dir_output,
                                            file_prefix,
                                            data_type,
                                            data_key,
                                            str(run))

            if os.path.exists(file_data_cache + '.gz'):
                file_data_cache = file_data_cache + '.gz'
            if not os.path.exists(file_data_cache):
                print('not cached', file_data_cache)
                data = utils.pkl_load(file_path)
                results_per_iteration = []
                data_iterations = get_learning_data(data_type,
                                                    data,
                                                    substring_exclude)
                data_ref = get_ref_data(data)
                for data_iteration in data_iterations:
                    results_per_iteration.append(
                        callback_fn(data_iteration, data_ref))
                results_per_iteration = np.array(results_per_iteration)
                utils.pkl_save(file_data_cache, results_per_iteration)
            else:
                results_per_iteration = utils.pkl_load(
                    file_data_cache)
            results_per_run.append(results_per_iteration)

        results_per_run = np.vstack(results_per_run)

        results.data[file_prefix] = results_per_run
        results.mean[file_prefix] = np.mean(results_per_run, axis=0)
        results.std[file_prefix] = np.std(results_per_run, axis=0)
        # We only need the template params file for one of the runs
        # The only change between runs is the random seed.
        # We always load the first run which has to be be available.
        results.params[file_prefix] = params

    return results


def compute_mean_abs_tracking_errors_q(dir_output, data_type, runs):
    def callback(data_iteration, data_ref):
        return np.mean(
            np.abs(
                data_iteration[defines.TRAJ_Q] -
                data_ref[defines.TRAJ_Q]))

    return _compute_mean(dir_output, data_type, 'q', runs, callback)


def compute_mean_abs_tracking_errors_qdd(dir_output, data_type, runs):
    def callback(data_iteration, data_ref):
        return np.mean(
            np.abs(
                data_iteration[defines.TRAJ_QDD_DESIRED] -
                data_iteration[defines.TRAJ_QDD_ACTUAL]))

    return _compute_mean(dir_output, data_type, 'qdd', runs, callback)


def compute_mean_abs_tau_feedback_learning(dir_output, data_type, runs):
    def callback(data_iteration, data_ref):
        return np.mean(
            np.abs(
                data_iteration[defines.TRAJ_TAU_FEEDBACK]))

    return _compute_mean(dir_output, data_type, 'tau_feedback_learning',
                         runs, callback)


def compute_mean_abs_tau_feedback_applied(dir_output, data_type, runs):
    def callback(data_iteration, data_ref):
        return np.mean(np.abs(data_iteration[defines.TRAJ_TAU] - (
            data_iteration[defines.TRAJ_TAU_FEEDFORWARD] +
            data_iteration[defines.TRAJ_TAU_INVERSE_DESIRED])))

    return _compute_mean(dir_output, data_type, 'tau_feedback_applied',
                         runs, callback)


def compute_mean_abs_tau(dir_output, data_type, runs):
    def callback(data_iteration, data_ref):
        return np.mean(np.abs(data_iteration[defines.TRAJ_TAU]))

    return _compute_mean(dir_output, data_type, 'tau', runs, callback)


def compute_mean_traj_q(dir_output, data_type, runs, substring_exclude=None):
    def callback(data_iteration, data_ref):
        return data_iteration[defines.TRAJ_Q]

    return _compute_mean(dir_output, data_type, 'traj_q', runs, callback,
                         substring_exclude)


def plots_save(dp_output,
               data_runs,
               name,
               average_params=None,
               exclude_by_key_value=None,
               ylim=None,
               yscale=None):
    data_runs = results_filter(data_runs, exclude_by_key_value)
    data_runs = results_average(data_runs, average_params)

    colors = {}
    colors['_indirect_'] = 'r'
    colors['_direct_'] = 'g'
    colors['_joined_'] = 'b'

    for parameter_unique, data in sorted(data_runs.items()):
        file_path = os.path.join(
            dp_output,
            replace_for_filepath(parameter_unique) + '_' + name + '.png')

        if os.path.exists(file_path):
            continue
        figure = plt.figure()
        axis = figure.add_subplot(111)
        if name.endswith('tau_feedback_applied'):
            axis.set_title('average feedback applied')
        elif name.endswith('q'):
            axis.set_title('average q error')
        elif name.endswith('qdd'):
            axis.set_title('average qdd error')
        for data_value in data:
            tmp_data = np.vstack(data_value.data)
            mean = np.mean(tmp_data, axis=0)
            std = np.std(tmp_data, axis=0)
            x = np.arange(mean.size)
            axis.plot(x,
                      mean,
                      label=data_value.data_type.replace('_', ''),
                      color=colors[data_value.data_type])
            axis.fill_between(x,
                              mean,
                              mean + std,
                              color=colors[data_value.data_type],
                              alpha=0.2)

        if yscale is not None:
            axis.set_yscale(yscale)
        if ylim is not None:
            plt.ylim(ylim)
        axis.legend(loc='upper right')
        plt.savefig(file_path)
        plt.close(figure)


def plots_save_traj(dp_output,
                    data_runs,
                    name,
                    average_params=None,
                    exclude_by_key_value=None,
                    ylim=None,
                    yscale=None):
    # data is a list with [([mean, std, results_per_run], data_type ), ...]
    data_runs = results_filter(data_runs, exclude_by_key_value)
    data_runs = results_average(data_runs, average_params)

    colors = {}
    colors['_indirect_'] = 'r'
    colors['_direct_'] = 'g'
    colors['_joined_'] = 'b'

    for parameter_unique, data in sorted(data_runs.items()):
        file_path = os.path.join(
            dp_output,
            replace_for_filepath(parameter_unique) + '_' + name + '.png')

        if os.path.exists(file_path):
            continue
        if name.endswith('tau_feedback_applied'):
            axis.set_title('average feedback applied')
        elif name.endswith('q'):
            axis.set_title('average q error')
        elif name.endswith('qdd'):
            axis.set_title('average qdd error')
        import ipdb
        ipdb.set_trace()

        figure = plt.figure()
        # first iteration
        axis = figure.add_subplot(311)
        for data_plot in data:
            data_combined = [np.expand_dims(tmp, axis=0)
                             for tmp in data_plot['data']]
            traj_mean = np.squeeze(np.mean(data_combined, axis=0), axis=0)
            traj_std = np.squeeze(np.mean(data_combined, axis=0), axis=0)
            axis = self._figure.add_subplot(
                3, 1, 1)
            axis.plot(np.arange(traj[:, 0].size),
                      traj_mean[:, 0],
                      color=color,
                      label=name)
            axis = self._figure.add_subplot(
                3, 1, 2)
            axis.plot(np.arange(traj[:, 1].size),
                      traj[:, 1],
                      color=color,
                      label=name)
            axis = self._figure.add_subplot(
                3, 1, 3)
            axis.plot(traj[:, 0],
                      traj[:, 1],
                      color=color,
                      label=name)

        # last iteration


def replace_for_filepath(string):
    return string.replace(' ', '').replace(
        ',', '_').replace('.', '_').replace('[', '').replace(']', '')


def results_filter(results, exclude_by_key_value=None):
    # results [result_for_data_type, ..., result_for_data_type]
    # Notice we have exactly the same keys for all of the datatypes.
    if exclude_by_key_value is not None:
        # We select the set of valid keys and remove all invalid ones
        # from the data.

        keys_ignore = []
        for key, params in sorted(results[0].params.items()):
            ignore = True
            for exclude_key, exclude_value in sorted(
                    exclude_by_key_value.items()):
                if str(params[exclude_key]) != str(exclude_value):
                    ignore = False
            if ignore:
                keys_ignore.append(key)

        # Implement the data subselection.
        for pos, result in enumerate(results):
            for key in result.keys():
                if key == 'data_type':
                    continue
                to_remove = result[key]
                for key_ignore in keys_ignore:
                    del to_remove[key_ignore]
                result[key] = to_remove
            results[pos] = result

    return results


def results_average(results, parameter_average=None):
    # results [result_for_data_type, ..., result_for_data_type]
    # Notice we have exactly the same keys for all of the datatypes.
    if parameter_average is None:
        parameter_average = ['RANDOM_SEED']
    else:
        parameter_average.append('RANDOM_SEED')
    keys_group = {}
    for key, params in sorted(results[0].params.items()):
        param_unique = {}
        for param_key, param_value in params.items():
            if param_key in parameter_average:
                continue
            if param_key == defines.EXPERIMENT_PATH:
                # This has to be ignored since it is always unique
                continue
            param_unique[param_key] = param_value
        params_str = '__'.join(
            [k + '_' + replace_for_filepath(str(v))
                for k, v in sorted(param_unique.items())])
        if params_str in keys_group:
            keys_group[params_str].append(key)
        else:
            keys_group[params_str] = [key]

    new_results = {}
    for key, values in keys_group.items():
        new_result = []
        for result in results:
            res = easydict.EasyDict()
            res.data_type = result.data_type
            res.data = [result.data[value] for value in values]
            new_result.append(res)
        new_results[key] = new_result

    return new_results


def main(argv=None):
    parser = argparse.ArgumentParser('Experiments.')

    parser.add_argument(
        '--dp_exp_root',
        type=str,
        help='The experiment root directory.')

    parser.add_argument(
        '--fp_config',
        type=str,
        help='The scenario we run the experiment with.')

    parser.add_argument(
        '--exp_type',
        default='exp',
        help='The type of experiment we run.')

    parser.add_argument(
        '--dp_output',
        type=str,
        default='/tmp',
        help='The output directory.')

    parser.add_argument(
        '--traj_q',
        action='store_true',
        help='Show the trajectory.')

    parser.add_argument(
        '--q',
        action='store_true',
        help='Compute q.')

    parser.add_argument(
        '--qdd',
        action='store_true',
        help='Compute qdd.')

    parser.add_argument(
        '--tau_feedback_learning',
        action='store_true',
        help='Compute tau_feedback.')

    parser.add_argument(
        '--tau_feedback_applied',
        action='store_true',
        help='Compute tau_feedback.')

    parser.add_argument(
        '--tau',
        action='store_true',
        help='Compute tau.')

    parser.add_argument(
        '--exclude_by_key_value',
        type=str,
        help='Exclude by key value (yaml str)')

    parser.add_argument(
        '--average_params',
        type=str,
        help='Name parameters which should be averaged (yaml str)')

    parser.add_argument(
        '--plot',
        action='store_true',
        help='Create the plots.')

    args, _ = parser.parse_known_args(argv)

    if args.exclude_by_key_value is not None:
        args.exclude_by_key_value = yaml.load(args.exclude_by_key_value)
    if args.average_params is not None:
        args.average_params = yaml.load(args.average_params)

    if args.dp_exp_root:
        runs = collect_runs_root(args.dp_exp_root)
    if args.fp_config:
        runs = collect_runs_single(args.fp_config, args.exp_type)

    data_types = ['_indirect_', '_direct_', '_joined_']
    try:

        if args.traj_q:
            cached = False
            file_path = None
            if args.dp_exp_root is not None:
                cached, file_path = get_file_path_cache(
                    args.dp_output, 'mean_traj_q')
            if not cached:
                mean_traj_q = []
                for data_type in data_types:
                    mean_traj_q.append(
                        compute_mean_traj_q(
                            args.dp_output, data_type, runs))

                mean_traj_q.append(
                    compute_mean_traj_q(
                        args.dp_output,
                        'ref_actual',
                        runs,
                        substring_exclude='feedback_learn'))
                mean_traj_q.append(
                    compute_mean_traj_q(
                        args.dp_output,
                        'act_desired',
                        runs,
                        substring_exclude='feedback_learn'))
                mean_traj_q.append(
                    compute_mean_traj_q(
                        args.dp_output,
                        'act_actual',
                        runs,
                        substring_exclude='feedback_learn'))
                mean_traj_q.append(
                    compute_mean_traj_q(
                        args.dp_output,
                        'act_actual_feedback',
                        runs,
                        substring_exclude='feedback_learn'))
            else:
                mean_traj_q = utils.pkl_load(file_path)
            print(file_path, cached)
            if file_path is not None and not cached:
                utils.pkl_save(file_path, mean_traj_q)

            if args.plot:
                plots_save_traj(args.dp_output,
                                mean_traj_q,
                                exclude_by_key_value=args.exclude_by_key_value,
                                average_params=args.average_params,
                                name='mean_traj_q')

        if args.q:
            cached = False
            file_path = None
            if args.dp_exp_root is not None:
                cached, file_path = get_file_path_cache(
                    args.dp_output, 'mean_abs_tracking_erros_q')
            if not cached:
                mean_abs_tracking_errors_q = []
                for data_type in data_types:
                    mean_abs_tracking_errors_q.append(
                        compute_mean_abs_tracking_errors_q(
                            args.dp_output, data_type, runs))
            else:
                mean_abs_tracking_errors_q = utils.pkl_load(file_path)
            print(file_path, cached)
            if file_path is not None and not cached:
                utils.pkl_save(file_path, mean_abs_tracking_errors_q)

            if args.plot:
                plots_save(args.dp_output,
                           mean_abs_tracking_errors_q,
                           exclude_by_key_value=args.exclude_by_key_value,
                           average_params=args.average_params,
                           name='mean_abs_tracking_erros_q',
                           yscale='log',
                           ylim=[0.0, 1000.0])

        if args.qdd:
            cached = False
            file_path = None
            if args.dp_exp_root is not None:
                cached, file_path = get_file_path_cache(
                    args.dp_output, 'mean_abs_tracking_erros_qdd')
            if not cached:
                mean_abs_tracking_errors_qdd = []
                for data_type in data_types:
                    mean_abs_tracking_errors_qdd.append(
                        compute_mean_abs_tracking_errors_qdd(
                            args.dp_output, data_type, runs))
            else:
                mean_abs_tracking_errors_qdd = utils.pkl_load(file_path)
            if file_path is not None and not cached:
                utils.pkl_save(file_path, mean_abs_tracking_errors_qdd)

            if args.plot:
                plots_save(args.dp_output,
                           mean_abs_tracking_errors_qdd,
                           exclude_by_key_value=args.exclude_by_key_value,
                           average_params=args.average_params,
                           name='mean_abs_tracking_erros_qdd',
                           yscale='log',
                           ylim=[50, 500.0])
        if args.tau_feedback_applied:
            cached = False
            file_path = None
            if args.dp_exp_root is not None:
                cached, file_path = get_file_path_cache(
                    args.dp_output, 'mean_abs_tracking_tau_feedback_applied')
            if not cached:
                mean_abs_tracking_tau_feedback_applied = []
                for data_type in data_types:
                    mean_abs_tracking_tau_feedback_applied.append(
                        compute_mean_abs_tau_feedback_applied(
                            args.dp_output, data_type, runs))
            else:
                mean_abs_tracking_tau_feedback_applied = utils.pkl_load(
                    file_path)
            if file_path is not None and not cached:
                utils.pkl_save(file_path,
                               mean_abs_tracking_tau_feedback_applied)
            if args.plot:
                plots_save(args.dp_output,
                           mean_abs_tracking_tau_feedback_applied,
                           exclude_by_key_value=args.exclude_by_key_value,
                           average_params=args.average_params,
                           name='mean_abs_tracking_tau_feedback_applied')
    except:
        utils.ipdb_exception()


if __name__ == '__main__':
    main()
