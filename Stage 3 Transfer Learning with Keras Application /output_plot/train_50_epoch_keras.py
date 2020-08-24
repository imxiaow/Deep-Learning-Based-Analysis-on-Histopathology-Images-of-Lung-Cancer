import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.optimizers import Adam
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

class TrainingPlot(keras.callbacks.Callback):
    #source: https://github.com/kapil-varshney/utilities/blob/master/training_plot/trainingplot.py
    def __init__(self, filename='/nfs/home/xwang/Keras_Transfer_Learning_June/output/training_plot_keras.png'):
        self.filename = filename

    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):

        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('accuracy'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_accuracy'))
        #loss: 7.6934 - accuracy: 0.7840 - val_loss: 7.6934 - val_accuracy: 0.7837

        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:

            N = np.arange(0, len(self.losses))

            # You can chose the style of your preference
            # print(plt.style.available) to see the available options
            plt.style.use("seaborn")

            # Plot train loss, train acc, val loss and val acc against epochs passed
            plt.figure()
            plt.plot(N, self.losses, label = "train_loss")
            plt.plot(N, self.acc, label = "train_acc")
            plt.plot(N, self.val_losses, label = "val_loss")
            plt.plot(N, self.val_acc, label = "val_acc")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(epoch))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            # Make sure there exists a folder called output in the current directory
            # or replace 'output' with whatever direcory you want to put in the plots
            plt.savefig(self.filename)
            plt.close()


if __name__ == "__main__":
    # Data loading 
    dataset_Keras_PATH = "/nfs/home/xwang/Keras_Transfer_Learning_June/Dataset_Keras_Folder/"
    kras_class_folder_path = "/nfs/home/xwang/Keras_Transfer_Learning_June/Dataset_Keras_Folder/class_kras/"
    nokras_class_folder_path = "/nfs/home/xwang/Keras_Transfer_Learning_June/Dataset_Keras_Folder/class_nokras/"
    
    # Create trainning dataset.
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(dataset_Keras_PATH, validation_split=0.3, subset="training", seed=2020, batch_size=200, image_size=(512, 512))
    # create validation dataset. 
    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(dataset_Keras_PATH, validation_split=0.3, subset="validation", seed=2020, batch_size=200,image_size=(512, 512))

    # Instantiate a base model and load pre-trained weights into it
    base_model = InceptionResNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(512, 512, 3)
        )

    # Freeze base model
    base_model.trainable = False

    # - Create a new model on top of the output of one (or several) layers from the base model.
    
    inputs = keras.Input(shape=(512, 512, 3))
    x = base_model(inputs, training=False)
    
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.5)(x)
    
    outputs = keras.layers.Dense(2, activation='softmax', name='softmax')(x)
    current_model = keras.Model(inputs, outputs)
    print(current_model.summary())

    #Cross-entropy is the default loss function to use for binary classification problems.
    #It is intended for use with binary classification where the target values are in the set {0, 1}.
    #loss_fn = keras.losses.BinaryCrossentropy()
    optimizer_adam = keras.optimizers.Adam(1e-3)#learning rate is default to 0.001
    
    # Create an instance of the TrainingPlot class with the filename.
    plot_losses = TrainingPlot()
    
    epochs = 50
    
    callbacks_plotloss = [
        plot_losses
    #keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
    ]
    
    current_model.compile(
        optimizer=optimizer_adam,
        loss="binary_crossentropy",
        metrics=["accuracy"],
        )
       
    #Configure the dataset for performance
    train_dataset = train_dataset.prefetch(buffer_size=200)
    validation_dataset = validation_dataset.prefetch(buffer_size=200)

    #Train the model using callback to the TrainingPlot class object
    current_model.fit(
        train_dataset, epochs=epochs, callbacks=callbacks_plotloss, validation_data=validation_dataset,
    )