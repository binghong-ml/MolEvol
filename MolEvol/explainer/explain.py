import keras
from keras_gcn import GraphConv
from keras.models import Model
from keras.engine.topology import Layer
from keras.layers import Dense, Input, Flatten, Add, Multiply, Lambda
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from keras import optimizers
import os
import numpy as np
import time

import tensorflow as tf
import pickle
import random

BATCH_SIZE = 1024
DATA_DIM = 12
MAX_ATOMS = 50
gumble_k = 12

tf.random.set_seed(0)
np.random.seed(0)
random.seed(0)


class Sample_Concrete(Layer):
    """
    Layer for sample Concrete / Gumbel-Softmax variables.

    """

    def __init__(self, tau0, k, **kwargs):
        self.tau0 = tau0
        self.k = k
        super(Sample_Concrete, self).__init__(**kwargs)

    def get_config(self):
        cfg = super().get_config()
        return cfg

    def call(self, logits, mask=None):
        # logits: [BATCH_SIZE, d]
        logits_ = K.expand_dims(logits, -2)  # [BATCH_SIZE, 1, d]

        batch_size = tf.shape(logits_)[0]
        d = tf.shape(logits_)[2]
        uniform = tf.random.uniform(shape=(batch_size, self.k, d),
                                    minval=np.finfo(tf.float32.as_numpy_dtype).tiny,
                                    maxval=1.0)

        gumbel = - K.log(-K.log(uniform))
        noisy_logits = (gumbel + logits_) / self.tau0
        samples = K.softmax(noisy_logits)
        samples = K.max(samples, axis=1)

        # Explanation Stage output.
        threshold = tf.expand_dims(tf.nn.top_k(logits, self.k, sorted=True)[0][:, -1], -1)
        discrete_logits = tf.cast(tf.greater_equal(logits, threshold), tf.float32)

        return K.in_train_phase(samples, discrete_logits)

    def compute_output_shape(self, input_shape):
        return input_shape


def configurate_model():
    node_label_input = keras.layers.Input(shape=(MAX_ATOMS,), dtype='int32')
    # node_feat_input = OnehotEmbedding(128)(node_one_hot_label_input)
    node_feat_input = keras.layers.Embedding(output_dim=128, input_dim=DATA_DIM)(
        node_label_input)
    adj_input = keras.layers.Input(shape=(MAX_ATOMS, MAX_ATOMS))

    gc1 = GraphConv(
        units=32,
        step_num=1,
    )([node_feat_input, adj_input])

    gc2 = GraphConv(
        units=32,
        step_num=1,
    )([gc1, adj_input])

    logits = Dense(1)(gc2)
    logits = K.squeeze(logits, -1)

    tau = 0.1
    samples = Sample_Concrete(tau, gumble_k, name='sample')(logits)

    selected_node_feat_input = Multiply()([gc2, K.expand_dims(samples, -1)])
    graph_feat_input = K.sum(selected_node_feat_input, axis=1) / gumble_k

    net = Dense(200, activation='relu', name='dense1',
                kernel_regularizer=regularizers.l2(1e-3))(graph_feat_input)
    net = BatchNormalization()(net)  # Add batchnorm for stability.
    net = Dense(200, activation='relu', name='dense2',
                kernel_regularizer=regularizers.l2(1e-3))(net)
    net = BatchNormalization()(net)

    preds = Dense(2, activation='softmax', name='dense4',
                  kernel_regularizer=regularizers.l2(1e-3))(net)

    model = Model(inputs=[node_label_input, adj_input], outputs=preds)
    return model, (node_label_input, adj_input), samples, logits


def get_pred_model(datatype='jnk3', k=gumble_k):
    model, model_input, samples, logits = configurate_model()
    model_path = 'models/explainer/L2X_k{}.hdf5.best'.format(k)
    model.load_weights(model_path, by_name=True)

    pred_logits_model = Model(model_input, logits)
    pred_logits_model.compile(loss=None,
                              optimizer='rmsprop',
                              metrics=[None])

    return pred_logits_model


prop4_model = get_pred_model('prop4')


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def softmax_with_temperature(x, t=50):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(t * (x - np.max(x)))
    return e_x / e_x.sum(axis=0)


from MolEvol.explainer.utils import get_graph


def L2X_explain(smiles):
    feat_np, adj_np = get_graph(smiles)
    adj_np = adj_np.reshape(1, MAX_ATOMS, MAX_ATOMS)
    num_atoms = feat_np.shape[0]
    tmp = np.zeros(MAX_ATOMS)
    tmp[:num_atoms] = feat_np
    feat_np = tmp.reshape(1, -1)

    logits_prop4 = prop4_model.predict([feat_np, adj_np], verbose=0, batch_size=1).reshape(-1)
    logits_prop4 = logits_prop4[:num_atoms]
    tmp = softmax_with_temperature(logits_prop4)
    return np.arange(num_atoms), tmp
