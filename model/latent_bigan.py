from __future__ import division, print_function, unicode_literals
import os
import yaml

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from model.activations import lrelu
from model.utils import sample_z

class LatentBiGAN:
    def __init__(self, name='LatentGAN', session=None):
        self.name = name
        self.params = self._load_params()
        self.z_dim = self.params['z_dim']
        self.data_dim = self.params['data_dim']
        self.batch_size = self.params['batch_size']
        self.n_epochs = self.params['epochs']

        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        self.X = tf.placeholder(tf.float32, shape=[None, self.data_dim])

        self.net_keep_prob = tf.placeholder_with_default(1.0, shape=[])

        self.is_training_pl = tf.placeholder(tf.bool, shape=[], name='is_training_pl')

        self._build_gan()
        self._build_optimiser()

        self.sess = session or tf.Session()
        self.saver = tf.train.Saver()

    def _load_params(self):
        '''Loads the parameters for every layer from a YAML file defined
        in `model/params.yml`
        Returns:
            params: Dictionary representing layer -> parameter mapping
        '''
        stream = open('model/params.yml', 'r')
        param_stream = yaml.load_all(stream)
        params = {}
        for param in param_stream:
            for k in param.keys():
                params[k] = param[k]

        return params

    def _build_gan(self):
        with tf.variable_scope(self.name) as scope:
            G = self._generator(self.z, self.is_training_pl, reuse=False)
            E = self._encoder(self.X, self.is_training_pl, reuse=False)
            D_real_prob, D_real_logits = self._discriminator(E, self.X, self.is_training_pl, reuse=False)
            D_gen_prob, D_gen_logits = self._discriminator(self.z, G, self.is_training_pl, reuse=True)

            G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_gen_logits), logits=D_gen_logits))
            E_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(D_real_logits), logits=D_real_logits))

            D_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_real_logits), logits=D_real_logits))
            D_gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(D_gen_logits), logits=D_gen_logits))
            D_loss = D_real_loss + D_gen_loss

            self.G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + '/G/')
            self.D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + '/D/')
            self.E_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + '/E/')

            self.G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name + '/G/')
            self.D_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name + '/D/')
            self.E_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name + '/E/')

        self.summary_op = tf.summary.merge([
            tf.summary.scalar('G_loss', tf.reduce_mean(G_loss)),
            tf.summary.scalar('D_loss', tf.reduce_mean(D_loss)),
            tf.summary.scalar('D_loss/real', tf.reduce_mean(D_real_loss)),
            tf.summary.scalar('D_loss/gen', tf.reduce_mean(D_gen_loss)),
        ])

        self.fake_sample = G
        self.encoding = E
        self.G_loss = G_loss
        self.D_loss = D_loss
        self.E_loss = E_loss

    def _build_optimiser(self):
        with tf.variable_scope('gan_optimiser'):
            with tf.control_dependencies(self.G_update_ops):
                self.G_train_op = tf.train.AdamOptimizer(**self.params['train']['generator']).\
                    minimize(self.G_loss, var_list=self.G_vars)
            with tf.control_dependencies(self.D_update_ops):
                self.D_train_op = tf.train.AdamOptimizer(**self.params['train']['discriminator'])\
                    .minimize(self.D_loss, var_list=self.D_vars)
            with tf.control_dependencies(self.E_update_ops):
                self.E_train_op = tf.train.AdamOptimizer(**self.params['train']['encoder'])\
                    .minimize(self.E_loss, var_list=self.E_vars)

    def _encoder(self, X, is_training, reuse=False):
        with tf.variable_scope('E', reuse=reuse) as scope:
            if reuse:
                scope.reuse_variables()
            net = X
            initializer = tf.random_normal_initializer(stddev=0.02)

            with tf.variable_scope('fc1'):
                net = tf.layers.dense(net, kernel_initializer=initializer,
                                    **self.params['encoder']['fc1'])

            with tf.variable_scope('fc2'):
                net = tf.layers.dense(net, kernel_initializer=initializer,
                                    **self.params['encoder']['fc2'])
                net = tf.layers.batch_normalization(net, training=is_training)

            with tf.variable_scope('fc3'):
                net = tf.layers.dense(net, kernel_initializer=initializer,
                                    **self.params['encoder']['fc3'])
                net = tf.layers.batch_normalization(net, training=is_training)

            with tf.variable_scope('fc_out'):
                z = tf.layers.dense(net, kernel_initializer=initializer,
                                    **self.params['encoder']['out'])

            return z 

    def _generator(self, z, is_training, reuse=False):
        with tf.variable_scope('G', reuse=reuse) as scope:
            if reuse:
                scope.reuse_variables()
            net = z
            initializer = tf.random_normal_initializer(stddev=0.02)

            with tf.variable_scope('fc1'):
                net = tf.layers.dense(net, kernel_initializer=initializer,
                                    **self.params['generator']['fc1'])
                net = tf.layers.batch_normalization(net, training=is_training)

            with tf.variable_scope('fc2'):
                net = tf.layers.dense(net, kernel_initializer=initializer,
                                    **self.params['generator']['fc2'])
                net = tf.layers.batch_normalization(net, training=is_training)

            with tf.variable_scope('fc3'):
                net = tf.layers.dense(net, kernel_initializer=initializer,
                                    **self.params['generator']['fc3'])
                net = tf.layers.batch_normalization(net, training=is_training)

            with tf.variable_scope('fc_out'):
                net = tf.layers.dense(net, kernel_initializer=initializer,
                                    **self.params['generator']['out'])

            return net

    def _discriminator(self, z, X, is_training, reuse=False):
        with tf.variable_scope('D', reuse=reuse):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            x = X
            with tf.variable_scope('fc1'):
                x = tf.layers.dense(x, **self.params['discriminator']['fc1'])
                x = tf.layers.batch_normalization(x, training=is_training)
                x = tf.layers.dropout(x, self.net_keep_prob, training=is_training)

            with tf.variable_scope('fc2'):
                x = tf.layers.dense(x, **self.params['discriminator']['fc2'])
                x = tf.layers.batch_normalization(x, training=is_training)
                x = tf.layers.dropout(x, self.net_keep_prob, training=is_training)

            with tf.variable_scope('fc3'):
                x = tf.layers.dense(x, **self.params['discriminator']['fc3'])
                x = tf.layers.batch_normalization(x, training=is_training)
                x = tf.layers.dropout(x, self.net_keep_prob, training=is_training)

            y = z
            with tf.variable_scope('fc4'):
                y = tf.layers.dense(y, **self.params['discriminator']['fc4']) 
                y = tf.layers.batch_normalization(y, training=is_training)
                y = tf.layers.dropout(y, self.net_keep_prob, training=is_training)

            with tf.variable_scope('fc5'):
                y = tf.layers.dense(y, **self.params['discriminator']['fc5']) 
                y = tf.layers.batch_normalization(y, training=is_training)
                y = tf.layers.dropout(y, self.net_keep_prob, training=is_training)

            x = tf.concat([x, y], axis=1)

            with tf.variable_scope('joint1'):
                x = tf.layers.dense(x, **self.params['discriminator']['joint1'])
                x = tf.layers.dropout(x, self.net_keep_prob, training=is_training)

            with tf.variable_scope('joint2'):
                x = tf.layers.dense(x, **self.params['discriminator']['joint2'])
                x = tf.layers.dropout(x, self.net_keep_prob, training=is_training)
            
            logits = tf.layers.dense(x, **self.params['discriminator']['logits'])
            prob = tf.sigmoid(logits)

            return prob, logits

    def train(self, data, restore_checkpoint=False, checkpoint_path='./checkpoints'):

        if restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
            self.saver.restore(self.sess, checkpoint_path)
        else:
            tf.global_variables_initializer().run(session=self.sess)

        X = data
        n_iterations_per_epoch = X.shape[0] // self.batch_size
        best_loss_val = np.infty

        summary_writer = tf.summary.FileWriter('train', graph=self.sess.graph)

        for epoch in range(self.n_epochs + 1):
            g_losses = []
            e_losses = []
            for i in range(n_iterations_per_epoch):
                half_batch = self.batch_size // 2

                z_D = sample_z(num=half_batch, dim=self.z_dim)
                start = i * half_batch 
                end = min((i+1) * half_batch, X.shape[0])
                X_D = X[start:end]

                if X_D.shape[0] < half_batch:
                    continue

                _, summary, d_loss = self.sess.run([
                    self.D_train_op,
                    self.summary_op,
                    self.D_loss
                ], feed_dict={self.X: X_D, self.z: z_D, self.is_training_pl: True, self.net_keep_prob: 0.2})

                z_G = sample_z(num=self.batch_size, dim=self.z_dim)
                _, g_loss = self.sess.run([
                    self.G_train_op,
                    self.G_loss
                ], feed_dict={self.z: z_G, self.is_training_pl: True, self.net_keep_prob: 0.2})

                g_losses.append(g_loss)
                summary_writer.add_summary(summary, global_step=epoch)

            g_loss = np.mean(g_losses)

            if g_loss < best_loss_val:
                self.saver.save(self.sess, checkpoint_path)
                best_loss_val = g_loss

        summary_writer.close()

    def generate_embedding(self, test_data):
        uninitialized_vars = []
        for var in tf.global_variables():
            try:
                self.sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninitialized_vars.append(var)

        self.sess.run(tf.variables_initializer(uninitialized_vars), feed_dict={self.test_data: test_data})

        epochs = self.params['embed_epochs']

        embedding_list = []
        for data_no, data in enumerate(test_data):

            data = np.expand_dims(data, axis=0)

            # Reset ano_Z
            self.sess.run(self.ano_z.initializer)

            for epoch in range(epochs + 1):
                _, ano_score, res_loss, dis_loss = self.sess.run([
                    self.ano_z_train_op,
                    self.anomaly_score,
                    self.res_loss,
                    self.dis_loss
                ], feed_dict={self.test_data: data})

            z_embed = self.sess.run([self.ano_z])
            z_embed = z_embed[0]
            embedding_list.append(z_embed)

            print('Done with data #{}'.format(data_no))

        embeddings = np.array(embedding_list)
        embeddings = np.squeeze(embeddings)
        np.save('embeddings', embeddings)
        embedding_var = tf.convert_to_tensor(embeddings, name='latent_embedding')

        config = projector.ProjectorConfig()

        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        embedding.metadata_path = os.path.join('test', 'metadata.tsv')

        summary_writer = tf.summary.FileWriter('test')

        projector.visualize_embeddings(summary_writer, config)

    def load(self, checkpoint_path='./checkpoints'):
        if tf.train.checkpoint_exists(checkpoint_path):
            self.saver.restore(self.sess, checkpoint_path)
            print('Checkpoint Restored.')
        else:
            tf.global_variables_initializer().run(session=self.sess)
            print('Could not restore. Variables re-initialised.')

    def generate(self, num=100):
        z = sample_z(num=num, dim=self.z_dim)

        fake_samples = self.sess.run([self.fake_sample], feed_dict={self.z: z})

        return fake_samples
