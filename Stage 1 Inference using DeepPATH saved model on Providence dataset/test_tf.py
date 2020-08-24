#%%
import tensorflow as tf
import tensorflow.contrib.slim as slim

def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


# %%
config = tf.ConfigProto(allow_soft_placement = True)
with tf.device('/gpu:0'):
    with tf.Session(config = config) as sess:
        saver = tf.train.import_meta_graph('/nfs6/deeppath_models/checkpoints/run1a_3D_classifier/model.ckpt-69000.meta')
        saver.restore(sess,tf.train.latest_checkpoint('/nfs6/deeppath_models/checkpoints/run1a_3D_classifier/'))
        model_summary()