import argparse
import copy
import itertools
import os

import jinja2
from inverse_dynamics import utils
from inverse_dynamics import defines


class ExperimentConfig(object):

    @classmethod
    def create_from_params(cls, params):
        return cls(params.experiment_name,
                   params.templates.fp_experiment_sh,
                   params.templates.fp_visualize_sh,
                   params.templates.fp_results_sh,
                   params.templates.fp_cluster_sh,
                   params.templates.fp_cluster_sub,
                   params.templates.fp_config,
                   params.experiments)

    def get_rel_file(self, fp, name):
        fp = utils.get_rel_project_path(fp)
        if not os.path.exists(fp):
            raise Exception(
                '{} does not exist {}.'.format(name, fp))
        return fp

    def __init__(self,
                 experiment_name,
                 fp_experiment_sh,
                 fp_visualize_sh,
                 fp_results_sh,
                 fp_cluster_sh,
                 fp_cluster_sub,
                 fp_config,
                 experiments):

        self._fp_experiment_sh = self.get_rel_file(fp_experiment_sh,
                                                   'fp_experiment_sh')
        self._fp_visualize_sh = self.get_rel_file(fp_visualize_sh,
                                                  'fp_visualize_sh')
        self._fp_results_sh = self.get_rel_file(fp_results_sh,
                                                'fp_results_sh')
        self._fp_cluster_sh = self.get_rel_file(fp_cluster_sh, 'fp_cluster_sh')
        self._fp_cluster_sub = self.get_rel_file(
            fp_cluster_sub, 'fp_cluster_sub')
        self._fp_config = self.get_rel_file(fp_config, 'fp_config')

        self._experiment_name = experiment_name
        self._experiments = experiments

        self._environment = jinja2.Environment(trim_blocks=True)

    def replace_for_filepath(self, string):
        return string.replace(' ', '').replace(
            ',', '_').replace('.', '_').replace('[', '').replace(']', '')

    def create_params_combination(self, params):
        params_list = []
        for key, value in params.items():
            params_list.append([{key: val} for val in value])
        return list(itertools.product(*params_list))

    def create_experiments_list(self):
        experiments = []
        for exp, value in sorted(self._experiments.items()):
            params_combination = self.create_params_combination(
                value.params_combination_setup)
            for param_group_name, param_group in sorted(
                    value.params_groups.items()):
                for param_combination in params_combination:
                    params = copy.deepcopy(param_group)
                    params['exp_type'] = value.exp_type
                    param_dict = {param.keys()[0]: param.values()[0]
                                  for param in param_combination}
                    params.update(param_dict)
                    random_seed = param_dict['RANDOM_SEED']
                    del param_dict['RANDOM_SEED']
                    params_str = '__'.join(
                        [k + '_' + self.replace_for_filepath(str(v))
                         for k, v in sorted(param_dict.items())])

                    exp_name = '__'.join(
                        [self._experiment_name, exp + '_' + param_group_name,
                         params_str, '{:02d}'.format(random_seed)])
                    experiments.append((exp_name, params))
        return experiments

    def create_experiment_structure(self):

        master_experiment_path = utils.get_rel_project_path(
            os.path.join(utils.get_dir_logs(), self._experiment_name))

        if os.path.exists(master_experiment_path):
            if not utils.query_yes_no(
                    'An experiment at the path ' +
                    master_experiment_path +
                    ' already exists. Still continue?',
                    default='yes'):
                print('We stop.')
                return
        else:
            os.makedirs(master_experiment_path)

        experiments = []
        results = []
        for exp_name, exp_params in self.create_experiments_list():
            experiment_script, fp_config = self.create_experiment_script(
                master_experiment_path, exp_name, exp_params)
            experiments.append(experiment_script)
            self.create_visualization_script(
                master_experiment_path, exp_name, exp_params, fp_config)
            results_script = self.create_result_script(
                master_experiment_path, exp_name, exp_params, fp_config)
            results.append(results_script)

        self.create_cluster_files(master_experiment_path,
                                  experiments,
                                  defines.EXPERIMENT_CLUSTER_EXP_SH,
                                  defines.EXPERIMENT_CLUSTER_EXP_SUB)

        self.create_cluster_files(master_experiment_path,
                                  results,
                                  defines.EXPERIMENT_CLUSTER_RES_SH,
                                  defines.EXPERIMENT_CLUSTER_RES_SUB)

    def create_cluster_files(self,
                             master_experiment_path,
                             commands,
                             cluster_sh,
                             cluster_sub):
        with open(self._fp_cluster_sh, 'r') as fi:
            template = self._environment.from_string(fi.read())

        fp_cluster_sh = utils.get_rel_project_path(
            os.path.join(master_experiment_path,
                         cluster_sh))

        with open(fp_cluster_sh, 'w') as fo:
            rel_project_dir = os.path.relpath(
                utils.get_dir_project(), os.path.dirname(
                    fp_cluster_sh))
            fo.write(template.render(
                commands=enumerate(commands),
                PROJECT_DIR=rel_project_dir))

        with open(self._fp_cluster_sub, 'r') as fi:
            template = self._environment.from_string(fi.read())

        cluster_sub = utils.get_rel_project_path(
            os.path.join(master_experiment_path,
                         cluster_sub))
        with open(cluster_sub, 'w') as fo:
            fo.write(template.render(
                cluster_script=cluster_sh,
                num_experiments=len(commands)))

    def create_result_script(self,
                             master_experiment_path,
                             exp_name,
                             exp_params,
                             fp_config):

        experiment_path = utils.get_rel_project_path(
            os.path.join(master_experiment_path, exp_name))
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)

        dp_output = os.path.join(master_experiment_path, 'results')

        results_script = utils.get_rel_project_path(
            os.path.join(experiment_path,
                         defines.EXPERIMENT_RESULTS_SH))

        with open(self._fp_results_sh, 'r') as fi:
            template = self._environment.from_string(fi.read())

        with open(results_script, 'w') as fo:
            rel_project_dir = os.path.relpath(
                utils.get_dir_project(), os.path.dirname(results_script))
            fo.write(template.render(
                FP_CONFIG=fp_config,
                EXP_TYPE=exp_params['exp_type'],
                DP_OUTPUT=dp_output,
                PROJECT_DIR=rel_project_dir))
        return results_script

    def create_visualization_script(self,
                                    master_experiment_path,
                                    exp_name,
                                    exp_params,
                                    fp_config):
        experiment_path = utils.get_rel_project_path(
            os.path.join(master_experiment_path, exp_name))
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)

        visualize_script = utils.get_rel_project_path(
            os.path.join(experiment_path,
                         defines.EXPERIMENT_VISUALIZE_SH))

        with open(self._fp_visualize_sh, 'r') as fi:
            template = self._environment.from_string(fi.read())

        with open(visualize_script, 'w') as fo:
            rel_project_dir = os.path.relpath(
                utils.get_dir_project(), os.path.dirname(visualize_script))
            fo.write(template.render(
                FP_CONFIG=fp_config,
                EXP_TYPE=exp_params['exp_type'],
                PROJECT_DIR=rel_project_dir))

    def create_experiment_script(self,
                                 master_experiment_path,
                                 exp_name,
                                 exp_params):

        experiment_path = utils.get_rel_project_path(
            os.path.join(master_experiment_path, exp_name))
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)

        with open(self._fp_config, 'r') as fi:
            template = self._environment.from_string(fi.read())

        fp_config = os.path.join(
            experiment_path, defines.EXPERIMENT_CONFIG_YAML)

        with open(fp_config, 'w') as fo:
            exp_params.update({'EXPERIMENT_PATH': experiment_path})
            fo.write(template.render(**exp_params))

        fp_template_params_yaml = os.path.join(
            experiment_path, defines.EXPERIMENT_TEMPLATE_PARAMS_YAML)

        utils.yaml_save(fp_template_params_yaml, dict(exp_params.items()))

        experiment_script = utils.get_rel_project_path(
            os.path.join(experiment_path,
                         defines.EXPERIMENT_EXPERIMENT_SH))

        with open(self._fp_experiment_sh, 'r') as fi:
            template = self._environment.from_string(fi.read())

        with open(experiment_script, 'w') as fo:
            rel_project_dir = os.path.relpath(
                utils.get_dir_project(), os.path.dirname(experiment_script))
            fo.write(template.render(
                FP_CONFIG=fp_config,
                EXP_TYPE=exp_params['exp_type'],
                PROJECT_DIR=rel_project_dir))

        return experiment_script, fp_config


def main(argv=None):
    parser = argparse.ArgumentParser('Experiments.')

    parser.add_argument(
        '--fp_config',
        default=os.path.join(os.path.dirname(__file__), 'config.yaml'),
        help='The scenario we run the experiment with.')

    parser.add_argument(
        '--overwrite',
        default=True,
        action='store_true',
        help='If we would rerun an experiment.')

    args, _ = parser.parse_known_args(argv)
    params = utils.params_load(args.fp_config)
    try:
        ec = ExperimentConfig.create_from_params(params)
        ec.create_experiment_structure()
    except:
        utils.ipdb_exception()


if __name__ == '__main__':
    main()
