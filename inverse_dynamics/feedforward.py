
import os
import sys
import numpy as np

import easydict
import tensorflow as tf
import tensorflow.contrib.slim as slim

from inverse_dynamics import factory

FACTORY = {}


def prelu(data, name_or_scope=None):
    with tf.variable_scope(
            name_or_scope,
            default_name='prelu',
            values=[data]):
        alphas = tf.get_variable(shape=data.get_shape().as_list()[-1:],
                                 initializer=tf.constant_initializer(0.01),
                                 name="alphas")

        return tf.nn.relu(data) + tf.multiply(
            alphas, (data - tf.abs(data))) * 0.5


class Interface(object):

    def __init__(self):
        pass

    def train(self, dataset_input, dataset_output, logdir):
        raise NotImplementedError

    def predict(self, q, qd, qdd):
        raise NotImplementedError

    def save(self, file_path_data):
        raise NotImplementedError


class FeedforwardNetworkInterface(Interface):

    @classmethod
    def create_from_params(cls, params, io_trans):
        return cls(
            params.fdf_layers,
            params.fdf_learning_rate,
            params.fdf_num_epochs,
            params.fdf_batch_size,
            params.fdf_val_every_n_steps,
            params.fdf_val_fraction,
            params.fdf_log_every_n_steps,
            params.fdf_summary_every_n_steps,
            params.fdf_random_seed,
            params.fdf_use_gpu,
            params.fdf_use_source,
            io_trans
        )

    def __init__(self,
                 layers,
                 learning_rate,
                 num_epochs,
                 batch_size,
                 val_every_n_steps,
                 val_fraction,
                 log_every_n_steps,
                 summary_every_n_steps,
                 random_seed,
                 use_gpu,
                 use_source,
                 io_trans):
        super(FeedforwardNetworkInterface, self).__init__()
        self._random_seed = random_seed
        self._index = None
        self._indices = None
        self._inference_op = None
        self._train_op = None
        self._layers = layers
        self._num_epochs = num_epochs
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._val_every_n_steps = val_every_n_steps
        self._val_fraction = val_fraction
        self._log_every_n_steps = log_every_n_steps
        self._summary_every_n_steps = summary_every_n_steps
        self._sess = None
        self._device = '/cpu:0'
        if use_gpu:
            self._device = '/gpu:0'
        self._use_source = use_source
        self._io_trans = io_trans

    def _construct_inference(self, s, q, qd, qdd, layers):
        raise NotImplementedError

    def _construct_loss(self, inference_op, tau):
        raise NotImplementedError

    def _construct_predict(self, inference_op):
        raise NotImplementedError

    def _update_index(self, traj_q):
        self._indices = np.arange(traj_q.shape[0])
        np.random.shuffle(self._indices)
        self._index = 0

    def _feed_dict(self,
                   traj_s,
                   traj_q,
                   traj_qd,
                   traj_qdd,
                   tau,
                   force_update=False):
        if self._index is None or force_update:
            self._update_index(traj_q)
        index_end = self._index + self._batch_size
        if index_end >= self._indices.size:
            self._update_index(traj_q)
            index_end = self._index + self._batch_size

        indices = self._indices[self._index:index_end]
        indices.sort()
        self._index = index_end
        feed_dict = {}
        if self._use_source:
            feed_dict[self._s_placeholder] = traj_s[indices, :].reshape(
                self._batch_size, -1)
        feed_dict[self._q_placeholder] = traj_q[indices, :].reshape(
            self._batch_size, -1)
        feed_dict[self._qd_placeholder] = traj_qd[indices, :].reshape(
            self._batch_size, -1)
        feed_dict[self._qdd_placeholder] = traj_qdd[indices, :].reshape(
            self._batch_size, -1)
        feed_dict[self._tau_placeholder] = tau[indices, :].reshape(
            self._batch_size, -1)

        return feed_dict

    def _create_index_list(self, size, fraction):
        count = 1. / fraction
        indices = []
        count_cur = 1 * count
        for pos in xrange(size):
            if pos > count_cur:
                indices.append(pos)
                count_cur += count
        return indices

    def _train_val_split(self, traj_s, traj_q, traj_qd, traj_qdd, traj_tau):
        indices_train = np.ones(traj_q.shape[0], dtype=bool)
        indices_train[
            self._create_index_list(traj_q.shape[0],
                                    self._val_fraction)] = False
        indices_val = ~indices_train
        train = easydict.EasyDict()
        train.traj_s = traj_s[indices_train]
        train.traj_q = traj_q[indices_train]
        train.traj_qd = traj_qd[indices_train]
        train.traj_qdd = traj_qdd[indices_train]
        train.traj_tau = traj_tau[indices_train]
        val = easydict.EasyDict()
        val.traj_s = traj_s[indices_val]
        val.traj_q = traj_q[indices_val]
        val.traj_qd = traj_qd[indices_val]
        val.traj_qdd = traj_qdd[indices_val]
        val.traj_tau = traj_tau[indices_val]
        return train, val

    def train(self, dataset_input, dataset_output, logdir):
        dataset_input = easydict.EasyDict(dataset_input)
        dataset_output = easydict.EasyDict(dataset_output)
        self._io_trans.init(dataset_input.traj_q,
                            dataset_input.traj_qd,
                            dataset_input.traj_qdd,
                            dataset_output.traj_tau)
        if not os.path.exists(logdir):
            os.makedirs(logdir)

        traj_q, traj_qd, traj_qdd = self._io_trans.input(
            dataset_input.traj_q,
            dataset_input.traj_qd,
            dataset_input.traj_qdd)
        traj_tau = self._io_trans.output_target(dataset_output['traj_tau'])

        train, val = self._train_val_split(dataset_input.traj_s,
                                           traj_q,
                                           traj_qd,
                                           traj_qdd,
                                           traj_tau)

        if self._inference_op is None:
            print('we construct the inference op')
            s = dataset_input.traj_s[0, :]
            q = traj_q[0, :]
            qd = traj_qd[0, :]
            qdd = traj_qdd[0, :]
            tau = traj_tau[0, :]

            tf.reset_default_graph()
            config = tf.ConfigProto()
            config.graph_options.optimizer_options.global_jit_level = (
                tf.OptimizerOptions.ON_1)

            self._sess = tf.Session(config=config)
            self._global_step = tf.contrib.framework.create_global_step()
            tf.set_random_seed(self._random_seed)
            self._inference_op = self._construct_inference(
                s, q, qd, qdd, self._layers)
            self._loss_op = self._construct_loss(self._inference_op, tau)
            self._predict_op = self._construct_predict(self._inference_op)
            optimizer_op = tf.train.AdamOptimizer(self._learning_rate)
            self._train_op = slim.learning.create_train_op(
                total_loss=self._loss_op,
                optimizer=optimizer_op,
                global_step=self._global_step,
                summarize_gradients=True)
            self._sess.run(tf.global_variables_initializer())
            self._sess.run(tf.local_variables_initializer())
            self._summary_writer = tf.summary.FileWriter(logdir)
            self._summary_writer.add_graph(tf.get_default_graph())
            self._saver = tf.train.Saver(max_to_keep=20)

        # For every new dataset we take the best possible model.
        # We do not use the model from before to make sure that we actually
        # achieve something good on the new dataset.
        self._val_loss = np.inf
        feed_dict = self._feed_dict(
            train.traj_s,
            train.traj_q,
            train.traj_qd,
            train.traj_qdd,
            train.traj_tau,
            force_update=True)

        epoch = 10 * train.traj_q.shape[0] / self._batch_size

        for step in xrange(
                self._num_epochs * (
                    train.traj_q.shape[0] / self._batch_size)):
            inputs = {}
            inputs['loss_np'] = self._train_op
            if step % self._log_every_n_steps == 0:
                inputs['inference_np'] = self._inference_op
                inputs['global_step_np'] = self._global_step
            if step % self._summary_every_n_steps == 0:
                inputs['summary'] = tf.summary.merge_all()
            if step % epoch == 0:
                inputs['global_step_np'] = self._global_step

            outputs = self._sess.run(inputs, feed_dict=feed_dict)
            feed_dict = self._feed_dict(train.traj_s,
                                        train.traj_q,
                                        train.traj_qd,
                                        train.traj_qdd,
                                        train.traj_tau)

            if step % self._log_every_n_steps == 0:
                print('step: ' + str(outputs['global_step_np']))
                print('loss: ' + str(outputs['loss_np']))

            if step % self._summary_every_n_steps == 0:
                self._summary_writer.add_summary(
                    outputs['summary'], global_step=outputs['global_step_np'])

            if step % epoch == 0:
                self._val_loss = self.val(val,
                                          outputs['global_step_np'],
                                          self._val_loss,
                                          logdir)
                feed_dict = self._feed_dict(train.traj_s,
                                            train.traj_q,
                                            train.traj_qd,
                                            train.traj_qdd,
                                            train.traj_tau,
                                            force_update=True)

        self._val_loss = self.val(val,
                                  self._sess.run(self._global_step),
                                  self._val_loss,
                                  logdir)
        self.restore_best_val(logdir)

    def restore_best_val(self, logdir):
        fp_model = os.path.join(logdir, 'model') + '-' + str(self._best_val)
        print('restore', fp_model)
        self._saver.restore(self._sess, fp_model)

    def val(self, validation_dataset, global_step, val_loss, logdir):
        total_loss = 0
        feed_dict = self._feed_dict(
            validation_dataset.traj_s,
            validation_dataset.traj_q,
            validation_dataset.traj_qd,
            validation_dataset.traj_qdd,
            validation_dataset.traj_tau,
            force_update=True)
        steps = (validation_dataset.traj_q.shape[0] / self._batch_size) + 1
        for step in xrange(steps):
            total_loss += self._sess.run(self._loss_op, feed_dict=feed_dict)
            feed_dict = self._feed_dict(
                validation_dataset.traj_s,
                validation_dataset.traj_q,
                validation_dataset.traj_qd,
                validation_dataset.traj_qdd,
                validation_dataset.traj_tau)

        total_loss /= steps

        print('val_loss', global_step, total_loss, val_loss)
        if total_loss < val_loss:
            # We have a better model, we actually save the model.
            self._best_val = global_step
            self._saver.save(self._sess,
                             os.path.join(logdir, 'model'),
                             global_step=global_step)

            return total_loss
        return val_loss

    def predict(self, q, qd, qdd):
        if self._inference_op is None:
            return np.zeros_like(q)

        q_int, qd_int, qdd_int = self._io_trans.input(q, qd, qdd)
        feed_dict = {
            self._q_placeholder: q_int.reshape(1, -1),
            self._qd_placeholder: qd_int.reshape(1, -1),
            self._qdd_placeholder: qdd_int.reshape(1, -1)
        }
        if self._use_source:
            feed_dict[self._s_placeholder] = np.zeros(
                1, dtype=np.float32).reshape(1, -1)

        return self._io_trans.output_predict(np.array(self._sess.run(
            self._predict_op, feed_dict=feed_dict)).flatten())

    def save(self, file_path_data):
        pass


class FeedforwardNetworkBasic(FeedforwardNetworkInterface):

    @classmethod
    def create_from_params(cls, params, io_trans):
        return cls(
            params.fdf_layers,
            params.fdf_learning_rate,
            params.fdf_num_epochs,
            params.fdf_batch_size,
            params.fdf_val_every_n_steps,
            params.fdf_val_fraction,
            params.fdf_log_every_n_steps,
            params.fdf_summary_every_n_steps,
            params.fdf_random_seed,
            params.fdf_use_gpu,
            params.fdf_use_source,
            io_trans
        )

    def __init__(self,
                 layers,
                 learning_rate,
                 num_epochs,
                 batch_size,
                 val_every_n_steps,
                 val_fraction,
                 log_every_n_steps,
                 summary_every_n_steps,
                 random_seed,
                 use_gpu,
                 use_source,
                 io_trans):
        super(FeedforwardNetworkBasic, self).__init__(
            layers,
            learning_rate,
            num_epochs,
            batch_size,
            val_every_n_steps,
            val_fraction,
            log_every_n_steps,
            summary_every_n_steps,
            random_seed,
            use_gpu,
            use_source,
            io_trans)

    def _construct_inference(self, s, q, qd, qdd, layers):
        inputs = []
        if self._use_source:
            self._s_placeholder = tf.placeholder(shape=(None, s.size),
                                                 dtype=tf.float32)
            inputs.append(self._s_placeholder)
        self._q_placeholder = tf.placeholder(shape=(None, q.size),
                                             dtype=tf.float32)
        inputs.append(self._q_placeholder)
        self._qd_placeholder = tf.placeholder(shape=(None, qd.size),
                                              dtype=tf.float32)
        inputs.append(self._qd_placeholder)
        self._qdd_placeholder = tf.placeholder(shape=(None, qdd.size),
                                               dtype=tf.float32)
        inputs.append(self._qdd_placeholder)

        result = []
        with tf.device(self._device):
            for pos in xrange(q.size):
                with tf.variable_scope('inference_op_' + str(pos),
                                       values=inputs):
                    net = tf.concat(
                        axis=1, values=inputs)

                    for layer in layers:
                        net = slim.layers.fully_connected(
                            net, layer, activation_fn=prelu)
                    result.append(slim.layers.fully_connected(
                        net, 1, activation_fn=None))
        return result

    def _construct_loss(self, inference_op, tau):
        self._tau_placeholder = tf.placeholder(shape=(None, tau.size),
                                               dtype=tf.float32)
        result = []
        for pos in xrange(tau.size):
            with tf.variable_scope('loss_op_' + str(pos),
                                   values=(inference_op +
                                           [self._tau_placeholder])):
                result.append(tf.losses.mean_squared_error(
                    inference_op[pos],
                    tf.reshape(self._tau_placeholder[:, pos], (-1, 1))))
                tf.summary.scalar('loss_' + str(pos), result[-1])
        return tf.add_n(result)

    def _construct_predict(self, inference_op):
        return inference_op

    def _update_index(self, traj_q):
        self._indices = np.arange(traj_q.shape[0])
        np.random.shuffle(self._indices)
        self._index = 0


factory.register(FeedforwardNetworkBasic, sys.modules[__name__])


def create_from_params(params, io_trans):
    if params.fdf_type not in FACTORY:
        raise Exception('The fdf_type {} is not available [{}]'.format(
            params.fdf_type, ','.join(FACTORY.keys())))

    return FACTORY[params.fdf_type].create_from_params(params, io_trans)
