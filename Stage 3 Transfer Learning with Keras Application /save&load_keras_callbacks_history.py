# https://stackoverflow.com/questions/49969006/save-and-load-keras-callbacks-history
# save and load <keras.callbacks.History>



"""
history_model_1 is a callback object. It contains all sorts of data and isn't serializable.

However, it contains a dictionnary with all the values that you actually want to save (cf your comment) :


import json
# Get the dictionary containing each metric and the loss for each epoch
history_dict = history_model_1.history
# Save it under the form of a json file
json.dump(history_dict, open(your_history_path, 'w'))

You can now access the value of the loss at the 50th epoch like this :

print(history_dict['loss'][49])

Reload it with
history_dict = json.load(open(your_history_path, 'r'))




You can use Pandas to save the history object as a CSV file.

import pandas as pd
pd.DataFrame.from_dict(history_model_1.history).to_csv('history.csv',index=False)


You can create a class so you will have the same structure and you can access in both cases with the same code.
import pickle
class History_trained_model(object):
    def __init__(self, history, epoch, params):
        self.history = history
        self.epoch = epoch
        self.params = params

with open(savemodel_path+'/history', 'wb') as file:
    model_history= History_trained_model(history.history, history.epoch, history.params)
    pickle.dump(model_history, file, pickle.HIGHEST_PROTOCOL)


then to access it:
with open(savemodel_path+'/history', 'rb') as file:
    history=pickle.load(file)

print(history.history)
"""


"""
#tf.keras.callbacks.History
http://man.hubwiz.com/docset/TensorFlow.docset/Contents/Resources/Documents/api_docs/python/tf/keras/callbacks/History.html
https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History#methods

"""