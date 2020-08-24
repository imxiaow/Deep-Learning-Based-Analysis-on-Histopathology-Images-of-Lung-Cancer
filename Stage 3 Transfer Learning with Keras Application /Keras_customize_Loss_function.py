# Make a custom loss function in keras
# https://stackoverflow.com/questions/45961428/make-a-custom-loss-function-in-keras

"""
There are two steps in implementing a parameterized custom loss function in Keras. 
First, writing a method for the coefficient/metric. 
Second, writing a wrapper function to format things the way Keras needs them to be.

1. It's actually quite a bit cleaner to use the Keras backend instead of tensorflow directly for simple custom loss functions like DICE. 
    Here's an example of the coefficient implemented that way:

    import keras.backend as K
    def dice_coef(y_true, y_pred, smooth, thresh):
        y_pred = y_pred > thresh
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

2. Now for the tricky part. 
    Keras loss functions must only take (y_true, y_pred) as parameters. 
    So we need a separate function that returns another function.


    def dice_loss(smooth, thresh):
          def dice(y_true, y_pred)
        return -dice_coef(y_true, y_pred, smooth, thresh)
    return dice



Finally, you can use it as follows in Keras compile.
# build model 
model = my_model()
# get the loss function
model_dice = dice_loss(smooth=1e-5, thresh=0.5)
# compile model
model.compile(loss=model_dice)

"""


# Custom loss function in Keras
# https://stackoverflow.com/questions/43818584/custom-loss-function-in-keras

"""
All you have to do is define a function for that, using keras backend functions for calculations. 
The function must take the true values and the model predicted values.
Now, since I'm not sure about what are g, q, x an y in your function, I'll just create a basic example here without caring about what it means or whether it's an actual useful function:

    import keras.backend as K
    def customLoss(yTrue,yPred):
        return K.sum(K.log(yTrue) - K.log(yPred))

All backend functions can be seen here: https://keras.io/backend/#backend-functions

After that, compile your model using that function instead of a regular one:
model.compile(loss=customLoss, optimizer = .....)
"""


# Keras Backend utilities
# https://keras.io/api/utils/backend_utils/




# Custom loss function in Keras based on the input data
# https://stackoverflow.com/questions/55445712/custom-loss-function-in-keras-based-on-the-input-data

"""
1. You can pass your input tensor as an argument to the custom loss wrapper function.

    def custom_loss(i):
    
        def loss(y_true, y_pred):
            return K.mean(K.square(y_pred - y_true), axis=-1) + something with i...
        return loss

    def baseline_model():
        # create model
        i = Input(shape=(5,))
        x = Dense(5, kernel_initializer='glorot_uniform', activation='linear')(i)
        o = Dense(1, kernel_initializer='normal', activation='linear')(x)
        model = Model(i, o)
        model.compile(loss=custom_loss(i), optimizer=Adam(lr=0.0005))
        return model

2. You can pad your label with extra data columns from input and write a custom loss. 
    This is helpful if you just want one/few feature column(s) from your input.

    def custom_loss(data, y_pred):
    
        y_true = data[:, 0]
        i = data[:, 1]
        return K.mean(K.square(y_pred - y_true), axis=-1) + something with i...


    def baseline_model():
        # create model
        i = Input(shape=(5,))
        x = Dense(5, kernel_initializer='glorot_uniform', activation='linear')(i)
        o = Dense(1, kernel_initializer='normal', activation='linear')(x)
        model = Model(i, o)
        model.compile(loss=custom_loss, optimizer=Adam(lr=0.0005))
        return model

    model.fit(X, np.append(Y_true, X[:, 0], axis =1), batch_size = batch_size, epochs=90, shuffle=True, verbose=1)

"""



# On Writing Custom Loss Functions in Keras
# https://medium.com/@yanfengliux/on-writing-custom-loss-functions-in-keras-e04290dd7a96