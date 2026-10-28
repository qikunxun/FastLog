import sys
import tensorflow as tf
import numpy as np
import pickle as pkl
from model_neurallp_tf import Learner
from_model_ckpt = sys.argv[1]
best_ckpt = sys.argv[2]

class Option(object):
    def __init__(self, d):
        self.__dict__ = d

config = {}
with open('{}/option.txt'.format(from_model_ckpt)) as fd:
    for line in fd:
        if not line: continue
        key, val = line.strip().split(', ')
        if '/' in val or '_' in val:
            config[key] = val
        else:
            config[key] = eval(val)


option = Option(config)
learner = Learner(option)
saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = False
config.allow_soft_placement = True
with tf.Session(config=config) as sess:
    tf.set_random_seed(1234)
    sess.run(tf.global_variables_initializer())
    print("Session initialized.")
    saver.restore(sess, '{}/ckpt/model-{}'.format(from_model_ckpt, best_ckpt))
    variables = tf.trainable_variables()
    atts, rnn_outputs = learner.get_attentions_given_queries(sess, np.arange(0, 24))
    atts = np.array(atts)
    var_list = []
    var_names = []
    for v in variables:
        print(v.name, v.shape)
        var_list.append(tf.get_default_graph().get_tensor_by_name(v.name))
        var_names.append(v.name)
    query_embedding_params = sess.run([learner.query_embedding_params])
    vars = sess.run(var_list)
    vars = zip(var_names, vars)
    weights = {'weights': atts}
    for (n, v) in vars:
        weights[n] = v
    np.save('{}/ckpt/model.weights.npy'.format(from_model_ckpt), weights)
    

    