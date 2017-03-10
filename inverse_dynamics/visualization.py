

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt  # noqa

from inverse_dynamics import defines


class Trajectory2d(object):

    def __init__(self, traj_key):
        self._figure = plt.figure()
        self._axis = self._figure.add_subplot(211)
        self._colors = ['b', 'g', 'r']
        self._plot = False
        self._traj_key = traj_key

    def _get_color(self, pos):
        if pos >= len(self._colors):
            print('We are out of predefined colors')
            return None
        return self._colors[pos]

    def plot(self, data):
        self.plot_first_last(data)

    def plot_first_last(self, data):
        pos = 0

        value_first = None
        value_last = None
        key_first = None
        key_last = None
        to_plot = []
        for key, value in sorted(data.items()):
            if value is None:
                continue

            if 'learn' not in key:
                to_plot.append((key, value))
                continue
            if self._traj_key in value:
                if value_first is None:
                    value_first = value
                    key_first = key
                value_last = value
                key_last = key

        self.plot_traj(value_first[self._traj_key],
                       self._get_color(0),
                       name=key_first)
        self.plot_traj(value_last[self._traj_key],
                       self._get_color(1),
                       name=key_last)
        for pos, (key, value) in enumerate(to_plot):
            if self._traj_key not in value:
                continue
            self.plot_traj(value[self._traj_key],
                           self._get_color(pos + 2),
                           name=key)
        self._axis.legend(loc='upper right')
        if pos > 0:
            self._plot = True

    def plot_traj(self, trajectory, color, name):
        traj = np.array(trajectory)
        self._axis = self._figure.add_subplot(
            3, 1, 1)
        self._axis.plot(np.arange(traj[:, 0].size),
                        traj[:, 0],
                        color=color,
                        label=name)
        self._axis = self._figure.add_subplot(
            3, 1, 2)
        self._axis.plot(np.arange(traj[:, 1].size),
                        traj[:, 1],
                        color=color,
                        label=name)
        self._axis = self._figure.add_subplot(
            3, 1, 3)
        self._axis.plot(traj[:, 0],
                        traj[:, 1],
                        color=color,
                        label=name)

    def show(self):
        if not self._plot:
            print('There is nothing to plot.')
            return
        plt.show(block=False)


class Trajectory1d(object):

    def __init__(self, traj_key):
        self._figure = plt.figure()
        self._colors = ['b', 'g', 'r']
        self._plot = False
        self._traj_key = traj_key
        self._axis = self._figure.add_subplot(2, 1, 1)
        self._axis.set_title(traj_key)

    def _get_color(self, pos):
        if pos >= len(self._colors):
            print('We are out of predefined colors')
            return None
        return self._colors[pos]

    def plot(self, data):
        self.plot_first_last(data)

    def plot_first_last(self, data):
        pos = 0

        value_first = None
        value_last = None
        key_first = None
        key_last = None
        to_plot = []
        for key, value in sorted(data.items()):
            if value is None:
                continue

            if 'learn' not in key:
                to_plot.append((key, value))
                continue
            if self._traj_key in value:
                if value_first is None:
                    value_first = value
                    key_first = key
                value_last = value
                key_last = key

        self.plot_traj(value_first[self._traj_key],
                       self._get_color(0),
                       name=key_first)
        self.plot_traj(value_last[self._traj_key],
                       self._get_color(1),
                       name=key_last)
        for pos, (key, value) in enumerate(to_plot):
            if self._traj_key not in value:
                continue
            self.plot_traj(value[self._traj_key],
                           self._get_color(pos + 2),
                           name=key)
        self._axis.legend(loc='upper right')
        if pos > 0:
            self._plot = True

    def plot_traj(self, trajectory, color, name):
        traj = np.array(trajectory)
        self._axis = self._figure.add_subplot(
            2, 1, 1)
        self._axis.plot(np.arange(traj[:, 0].size),
                        traj[:, 0],
                        color=color,
                        label=name)
        self._axis = self._figure.add_subplot(
            2, 1, 2)
        self._axis.plot(np.arange(traj[:, 1].size),
                        traj[:, 1],
                        color=color,
                        label=name)

    def show(self):
        if not self._plot:
            print('There is nothing to plot.')
            return
        plt.show(block=False)


class Trajectory1dError(object):

    def __init__(self, traj_key_actual, traj_key_desired):
        self._figure = plt.figure()
        self._colors = ['b', 'g', 'r']
        self._plot = False
        self._traj_key_actual = traj_key_actual
        self._traj_key_desired = traj_key_desired
        self._axis = self._figure.add_subplot(2, 1, 1)

    def _get_color(self, pos):
        if pos >= len(self._colors):
            print('We are out of predefined colors')
            return None
        return self._colors[pos]

    def plot(self, data):
        pos = 0
        for key, value in sorted(data.items()):
            if value is None:
                continue
            if 'learn' not in key:
                continue
            if (self._traj_key_actual in value and
                    self._traj_key_desired in value):
                actual = value[self._traj_key_actual]
                desired = value[self._traj_key_desired]
                diff = actual - desired
                error = np.abs(diff)
                self.plot_traj(error,
                               self._get_color(pos),
                               name=key)
                pos += 1
        self._axis.legend(loc='upper left')
        if pos > 0:
            self._plot = True

    def plot_traj(self, trajectory, color, name):
        traj = np.array(trajectory)
        self._axis = self._figure.add_subplot(
            2, 1, 1)
        self._axis.plot(np.arange(traj[:, 0].size),
                        traj[:, 0],
                        color=color,
                        label=name)
        self._axis = self._figure.add_subplot(
            2, 1, 2)
        self._axis.plot(np.arange(traj[:, 1].size),
                        traj[:, 1],
                        color=color,
                        label=name)

    def show(self):
        if not self._plot:
            print('There is nothing to plot.')
            return
        plt.show(block=False)


class Dataset(object):

    def __init__(self):
        self._colors = ['b', 'g', 'r']
        self._plot = False

    def _get_color(self, pos):
        if pos >= len(self._colors):
            print('We are out of predefined colors')
            return None
        return self._colors[pos]

    def plot(self, data):
        pos = 0
        result_key = None
        for key, value in sorted(data.items()):

            if value is None:
                continue
            if 'learn' not in key:
                continue

            if defines.DATASET_INPUT in value:
                dataset_input = value[defines.DATASET_INPUT]
                dataset_output = value[defines.DATASET_OUTPUT]
                result_key = key
                pos += 1

        self._figure = plt.figure()
        self._axis = self._figure.add_subplot(2, 1, 1, projection='3d')
        print(result_key)
        self.plot_traj(dataset_input,
                       dataset_output,
                       name=result_key)
        self._axis.legend(loc='upper left')
        plt.show(block=False)
        if pos > 0:
            self._plot = True

    def plot_traj(self, dataset_input, dataset_output, name):
        xs = dataset_input[defines.TRAJ_Q][:100]
        ys = dataset_input[defines.TRAJ_QD][:100]
        zs = dataset_input[defines.TRAJ_QDD][:100]
        zs = dataset_output[defines.TRAJ_TAU][:100]
        tau = dataset_output[defines.TRAJ_TAU][:100]

        if dataset_input[defines.TRAJ_Q].shape[0] > 1000:
            xs = []
            ys = []
            zs = []
            tau = []
            for pos in xrange(100):
                xs.append(dataset_input[defines.TRAJ_Q][pos, :])
                xs.append(dataset_input[defines.TRAJ_Q][pos + 1000, :])
                ys.append(dataset_input[defines.TRAJ_QD][pos, :])
                ys.append(dataset_input[defines.TRAJ_QD][pos + 1000, :])
                zs.append(dataset_input[defines.TRAJ_QDD][pos, :])
                zs.append(dataset_input[defines.TRAJ_QDD][pos + 1000, :])
                tau.append(dataset_output[defines.TRAJ_TAU][pos, :])
                tau.append(dataset_output[defines.TRAJ_TAU][pos + 1000, :])
            xs = np.array(xs)
            ys = np.array(ys)
            zs = np.array(zs)
            tau = np.array(tau)

        # tau_thresh = 4000
        for pos in xrange(2):
            col = tau[:, pos]
            col = np.arange(col.shape[0])
            color = matplotlib.cm.hsv(col)
            colmap = matplotlib.cm.ScalarMappable(cmap=matplotlib.cm.hsv)
            colmap.set_array(col)

            self._axis = self._figure.add_subplot(
                2, 1, pos + 1, projection='3d')
            self._axis.scatter(xs[:, pos],
                               ys[:, pos],
                               zs[:, pos],
                               color=color,
                               label=name)
            self._figure.colorbar(colmap)

    def show(self):
        if not self._plot:
            print('There is nothing to plot.')
            return
        plt.show(block=False)


def show():
    plt.show(block=True)
