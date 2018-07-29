from __future__ import division, print_function, unicode_literals
import os
import yaml

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from model.activations import lrelu
from model.utils import sample_z

class LatentGAN:
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
            G = self._generator(self.z, reuse=False)
            D_real_prob, D_real_logits = self._discriminator(self.X, reuse=False)
            D_gen_prob, D_gen_logits = self._discriminator(G, reuse=True)

            G_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(D_gen_logits), logits=D_gen_logits)
            D_real_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(D_real_logits), logits=D_real_logits)
            D_gen_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_gen_logits), logits=D_gen_logits)
            D_loss = D_real_loss + D_gen_loss

            self.G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + '/G/')
            self.D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + '/D/')

            self.G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name + '/G/')
            self.D_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name + '/D/')

        self.summary_op = tf.summary.merge([
            tf.summary.scalar('G_loss', tf.reduce_mean(G_loss)),
            tf.summary.scalar('D_loss', tf.reduce_mean(D_loss)),
            tf.summary.scalar('D_loss/real', tf.reduce_mean(D_real_loss)),
            tf.summary.scalar('D_loss/gen', tf.reduce_mean(D_gen_loss)),
        ])

        self.fake_sample = G
        self.G_loss = G_loss
        self.D_loss = D_loss

    def _build_optimiser(self):
        with tf.variable_scope('gan_optimiser'):
            with tf.control_dependencies(self.G_update_ops):
                self.G_train_op = tf.train.AdamOptimizer(**self.params['train']['generator']).\
                    minimize(self.G_loss, var_list=self.G_vars)
            with tf.control_dependencies(self.D_update_ops):
                # self.D_train_op = tf.train.AdamOptimizer(**self.params['train']['discriminator'])\
                #     .minimize(self.D_loss, var_list=self.D_vars)
                self.D_train_op = tf.train.MomentumOptimizer(**self.params['train']['discriminator']).\
                    minimize(self.D_loss, var_list=self.D_vars)

    def build_gan_embedding(self, lambda_ano=0.1):
        with tf.variable_scope(self.name):
            self.test_data = tf.placeholder(tf.float32, shape=[None, self.data_dim], name='ano_X')

            with tf.variable_scope('AnoD'):
                self.ano_z = tf.get_variable('ano_z', shape=[1, self.z_dim], dtype=tf.float32,
                                             initializer=tf.random_normal_initializer(stddev=0.5,
                                                                                      dtype=tf.float32))

            self.ano_G = self._generator(self.ano_z, reuse=True)

            self.res_loss = tf.reduce_sum(tf.abs(self.test_data - self.ano_G))

            self.d_feature_test = self._discriminator_feature_extractor(self.test_data)
            self.d_feature_z = self._discriminator_feature_extractor(self.ano_G)
            self.dis_loss = tf.reduce_mean(tf.abs(self.d_feature_test - self.d_feature_z))

            self.anomaly_score = (1. - lambda_ano) * self.res_loss + lambda_ano * self.dis_loss

            ano_z_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + '/AnoD/')

            ano_z_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name + '/AnoD/')

            with tf.control_dependencies(ano_z_update_ops):
                self.ano_z_train_op = tf.train.AdamOptimizer(learning_rate=1e-2, beta1=0.9).\
                    minimize(self.anomaly_score, var_list=ano_z_vars)

    def _encoder(self, X, reuse=False):
        with tf.variable_scope('E', reuse=reuse) as scope:
            if reuse:
                scope.reuse_variables()
            net = X
            initializer = tf.random_normal_initializer(stddev=0.02)
            net = tf.layers.dense(net, kernel_initializer=initializer,
                                  **self.params['encoder']['fc1'])
            net = tf.layers.dropout(net, self.net_keep_prob, training=True)
            net = tf.layers.dense(net, kernel_initializer=initializer,
                                  **self.params['encoder']['fc2'])
            net = tf.layers.dropout(net, self.net_keep_prob, training=True)
            net = tf.layers.dense(net, kernel_initializer=initializer,
                                  **self.params['encoder']['fc3'])
            net = tf.layers.dropout(net, self.net_keep_prob, training=True)
            net = tf.layers.dense(net, kernel_initializer=initializer,
                                  **self.params['encoder']['out'])

            return net

    def _generator(self, z, reuse=False):
        with tf.variable_scope('G', reuse=reuse) as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            net = z
            initializer = tf.random_normal_initializer(stddev=0.02)
            net = tf.layers.dense(net, kernel_initializer=initializer,
                                  **self.params['generator']['fc1'])
            net = tf.layers.dropout(net, self.net_keep_prob, training=True)
            net = tf.layers.dense(net, kernel_initializer=initializer,
                                  **self.params['generator']['fc2'])
            net = tf.layers.dropout(net, self.net_keep_prob, training=True)
            net = tf.layers.dense(net, kernel_initializer=initializer,
                                  **self.params['generator']['fc3'])
            net = tf.layers.dropout(net, self.net_keep_prob, training=True)
            net = tf.layers.dense(net, kernel_initializer=initializer,
                                  **self.params['generator']['out'])

            return net

    def _discriminator(self, z, X, reuse=False):
        with tf.variable_scope('D', reuse=reuse):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            x = X
            x = tf.layers.dense(x, **self.params['discriminator']['fc1'])
            x = tf.layers.dropout(x, self.net_keep_prob)
            x = tf.layers.dense(x, **self.params['discriminator']['fc2'])
            x = tf.layers.dropout(x, self.net_keep_prob)
            x = tf.layers.dense(x, **self.params['discriminator']['fc3'])
            x = tf.layers.dropout(x, self.net_keep_prob)

            y = z
            y = tf.layers.dense(y, **self.params['discriminator']['fc4']) 
            y = tf.layers.dropout(y, self.net_keep_prob, training=True)
            y = tf.layers.dense(y, **self.params['discriminator']['fc5']) 
            y = tf.layers.dropout(y, self.net_keep_prob, training=True)

            x = tf.concat([x, y], axis=1)
            x = tf.layers.dense(x, **self.params['discriminator']['joint1'])
            x = tf.layers.dropout(x, self.net_keep_prob)
            x = tf.layers.dense(x, **self.params['discriminator']['joint2'])
            x = tf.layers.dropout(x, self.net_keep_prob)
            
            logits = tf.layers.dense(x, **self.params['discriminator']['logits'])
            prob = tf.sigmoid(logits)

            return prob, logits

    def _discriminator_feature_extractor(self, X, reuse=True):
        with tf.variable_scope('D', reuse=reuse):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            net = X
            net = tf.layers.dense(net, **self.params['discriminator']['fc1'])
            net = tf.layers.dropout(net, self.net_keep_prob)
            net = tf.layers.dense(net, **self.params['discriminator']['fc2'])
            net = tf.layers.dropout(net, self.net_keep_prob)
            net = tf.layers.dense(net, **self.params['discriminator']['fc3'])

            return net 

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
                ], feed_dict={self.X: X_D, self.z: z_D, self.net_keep_prob: 0.5})

                z_G = sample_z(num=self.batch_size, dim=self.z_dim)
                _, g_loss = self.sess.run([
                    self.G_train_op,
                    self.G_loss
                ], feed_dict={self.z: z_G, self.net_keep_prob: 0.5})

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
