import click

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from model.latent_bigan import LatentBiGAN 

@click.command()
@click.option('-d', '--data',
               help='Dataset for training. Should have format \
              (X_train, y_train, X_test, y_test, X_valid, y_valid).')
@click.option('--use-gpu/--no-use-gpu', default=False,
              help='Set this flag to enable GPU support')
@click.option('--cuda-devices', default='0, 1',
              help='CUDA Device indexes to expose to Tensorflow')
@click.option('--log-device-placement/--no-log-device-placement',
              default=False)
def main(data, use_gpu, cuda_devices, log_device_placement):
    data_raw = np.load(data)
    dataset_dict = dict(data_raw.items())
    X_train, X_test, y_test = dataset_dict.values()

    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(np.concatenate((X_train, X_test))) 
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    config = tf.ConfigProto()
    if use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices 
        config.gpu_options.allow_growth = True
        config.log_device_placement = log_device_placement
        config.allow_soft_placement = True

    g_gan = tf.Graph()
    with g_gan.as_default():
        with tf.Session(config=config) as sess:

            model = LatentBiGAN(session=sess)

            model.train(data=X_train)

    g_ano = tf.Graph()
    with g_ano.as_default():
        with tf.Session(config=config) as sess:

            model_embed = LatentBiGAN(session=sess)
            model_embed.build_gan_embedding()

            model_embed.load()

            model_embed.generate_embedding(X_test)

if __name__ == '__main__':
    main()