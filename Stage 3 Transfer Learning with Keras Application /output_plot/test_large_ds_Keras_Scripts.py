import tensorflow as tf
from tensorflow import keras

import os
import tempfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2

import pickle

mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

if __name__ == "__main__":
    # data loading 
    TLarge_dataset_Keras_PATH = "/mnt/app_hdd/scratch/keras_dataset/"
    
    # Create trainning dataset.
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(TLarge_dataset_Keras_PATH, validation_split=0.3, subset="training", batch_size=200, shuffle=False, image_size=(512, 512), label_mode='categorical')
    # Create validation dataset. 
    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(TLarge_dataset_Keras_PATH, validation_split=0.3, subset="validation", batch_size=200, shuffle=False, image_size=(512, 512), label_mode='categorical')

    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    
    # Open a strategy scope.
    with strategy.scope():
        # Everything that creates variables should be under the strategy scope.
        # In general this is only model construction & `compile()`.
        IMG_SHAPE = (512, 512, 3)
        # Create the base model from the pre-trained model 
        pretrained_model = InceptionResNetV2(
            input_shape=(512, 512, 3),
            include_top=False,
            weights='imagenet')

        # Freeze the convolutional base
        pretrained_model.trainable = False

        # Add a classification head
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        dropout_layer = keras.layers.Dropout(0.5)
        prediction_layer = tf.keras.layers.Dense(2, activation='softmax', name="softmax")
        
        # Now stack the feature extractor, and these two layers using a tf.keras.Sequential model:
        transfer_learning_model = tf.keras.Sequential([
            pretrained_model,
            global_average_layer,
            dropout_layer,
            prediction_layer
            ])
        
        print(transfer_learning_model.summary())
        
        #Compile the model
        transfer_learning_model.compile(
            optimizer=keras.optimizers.Adam(1e-3), 
            loss=keras.losses.CategoricalCrossentropy(), 
            metrics =keras.metrics.AUC())

    initial_epochs = 10
    validation_steps = 10

    # WARNING:tensorflow:Your input ran out of data; interrupting training.
    # Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 20 batches).

    loss0,accuracy0 = transfer_learning_model.evaluate(validation_dataset, steps = validation_steps)
    print("initial loss: {:.2f}".format(loss0))
    print("initial AUC: {:.2f}".format(accuracy0))

    history = transfer_learning_model.fit(
        train_dataset,
        epochs=initial_epochs,
        validation_data=validation_dataset)
    
    #save the history
    with open('/nfs/home/xwang/Keras_Transfer_Learning_June/test_large_ds_results/trainHistoryDict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    file_pi.close()

    # Save the entire model to a HDF5 file.
    # # The '.h5' extension indicates that the model should be saved to HDF5.
    transfer_learning_model.save('/nfs/home/xwang/Keras_Transfer_Learning_June/test_large_ds_results/transfer_learning_model_toy.h5') 



