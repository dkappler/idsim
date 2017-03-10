
import time
import sys
import os
import yaml
import easydict
try:
    import cPickle as pkl
except:
    import pickle as pkl

import gzip

from inverse_dynamics import defines


class Timer(object):

    def __init__(self, comment):
        self._comment = comment

    def __enter__(self):
        self.__start = time.time()

    def __exit__(self, type, value, traceback):
        self.__finish = time.time()
        print(
            self._comment,
            "duration in seconds:",
            self.duration_in_seconds())

    def duration_in_seconds(self):
        return self.__finish - self.__start


def params_load(file_path):
    params_dict = yaml_load(file_path)
    return easydict.EasyDict(params_dict)


def yaml_load(file_path):
    if not os.path.exists(file_path):
        raise Exception('No file exists at {}.'.format(file_path))
    with open(file_path, 'r') as fi:
        return yaml.load(fi)


def yaml_save(file_path, data, check_overwrite=False, check_create=False):
    if os.path.exists(file_path):
        if check_overwrite and not query_yes_no(
                'File {} exists, do you want to overwrite?'.format(file_path),
                'yes'):
            return
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        if check_create and not query_yes_no(
                'Output directory {} does not exist, '
                'do you want to create it?'.format(dir_path),
                'yes'):
            return
        os.makedirs(dir_path)
    with open(file_path, 'w') as fo:
        return yaml.dump(data, fo)


def pkl_load(file_path):
    if file_path.endswith('.gz'):
        with gzip.open(file_path, 'r') as fi:
            return pkl.load(fi)

    if os.path.exists(file_path + '.gz'):
        with gzip.open(file_path + '.gz', 'r') as fi:
            return pkl.load(fi)
    else:
        if not os.path.exists(file_path):
            raise Exception('No file exists at {}.'.format(file_path))
        with open(file_path, 'r') as fi:
            return pkl.load(fi)


def pkl_save(file_path,
             data,
             check_overwrite=False,
             check_create=False,
             compress=True):
    if compress:
        file_path += '.gz'
    if os.path.exists(file_path):
        if check_overwrite and not query_yes_no(
                'File {} exists, do you want to overwrite?'.format(file_path),
                'yes'):
            return
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        if check_create and not query_yes_no(
                'Output directory {} does not exist, '
                'do you want to create it?'.format(dir_path),
                'yes'):
            return
        os.makedirs(dir_path)
    if compress:
        with gzip.open(file_path, 'w') as fo:
            return pkl.dump(data, fo, protocol=pkl.HIGHEST_PROTOCOL)
    else:
        with open(file_path, 'w') as fo:
            return pkl.dump(data, fo)


def query_yes_no(question, default='yes'):
    """Ask a yes/no question via raw_input() and return their answer.

    'question' is a string that is presented to the user.
    'default' is the presumed answer if the user just hits <Enter>.
        It must be 'yes' (the default), 'no' or None (meaning
        an answer is required of the user).

    The 'answer' return value is True for 'yes' or False for 'no'.
    """
    valid = {'yes': True, 'y': True, 'ye': True,
             'no': False, 'n': False}
    if default is None:
        prompt = ' [y/n] '
    elif default == 'yes':
        prompt = ' [Y/n] '
    elif default == 'no':
        prompt = ' [y/N] '
    else:
        raise ValueError('Invalid default answer: {}.'.format(default))

    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write('Please respond with "yes" or "no" '
                             '(or "y" or "n").\n')


def ipdb_exception():
    import traceback
    exc_type, exc_value, exc_traceback = sys.exc_info()
    traceback.print_tb(exc_traceback, file=sys.stdout)
    traceback.print_exc(file=sys.stdout)
    traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)
    _, _, tb = sys.exc_info()
    try:
        import ipdb
        ipdb.post_mortem(tb)
    except:
        sys.exit(-1)


class ExponentialMovingAverage(object):

    def __init__(self, alpha, forgetting_rate=1.0):
        self._alpha = alpha
        self._forgetting_rate = forgetting_rate
        self._data = None

    def reset(self):
        self._data = None

    def update(self, data):
        if self._data is None:
            self._data = data
            return
        self._data = ((self._alpha) * data +
                      (1 - self._alpha) * self._forgetting_rate * self._data)

    @property
    def data(self):
        return self._data


def get_dir_project():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def get_rel_project_path(path):
    return os.path.relpath(path, get_dir_project())


def get_dir_logs():
    dir_project = get_dir_project()
    return os.path.join(dir_project, defines.LOGS)


def get_logdir(experiment_name):
    dir_project = get_dir_project()
    logdir = os.path.join(
        os.path.join(dir_project, defines.LOGS), experiment_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    return logdir
