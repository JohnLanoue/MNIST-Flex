#!/usr/bin/env python
# coding: utf-8

# # Assignment 6 
# ## Import the data 
# As seen before we can easielly import the data using built in dataset commands.  From what was in the previous assignment, MNIST is a list of 60000 handwritten images of handwritten numbers made of 28x28 images. Upon importing we can divide the datasets directly.  In order for my computer to process the data in decent time, we should thin out both datasets.  

# In[1]:


from tensorflow import keras
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_valid = X_train[:5000]/255.
X_train = X_train[5000:]/255.
X_test = X_test/255.
y_valid = y_train[:5000]
y_train = y_train[5000:]


# We are also going to want to scale the digits between 0 and 1.  
# 

# X_train = X_train/255.
# X_valid = X_valid/255. 
# X_test = X_test / 255.

# As means for validation of a successful import, lets have a look at an image. 

# In[2]:


import matplotlib.pyplot as plt
plt.imshow(X_train[0], cmap="binary")
plt.axis('off')
plt.show()


# The data agrees with my eyes, that's a 5.  With any luck we will get the same eyes with our  model. 

# In[3]:


y_train[0]


# # Building our Network
# We should start by evaluating the learning rate.  Using a gsd we will determin the proper parameter for the dataset. The object below includes two arrays: rates and losses.  A low learning rate will cause us to not pick up proper 'rules' for determining which digit is which - a high learning rate is a sure way to create overfit.  Losses are the gradieants in which the backpropaged weights are updated. Finally, the object is included with the Keras.backend.set_value.  

# In[4]:


K = keras.backend

class ExponentialLearningRate(keras.callbacks.Callback):
    def __init__(self, factor):
        self.factor = factor
        self.rates = []
        self.losses = []
    def on_batch_end(self, batch, logs):
        self.rates.append(K.get_value(self.model.optimizer.learning_rate))
        self.losses.append(logs["loss"])
        K.set_value(self.model.optimizer.learning_rate, self.model.optimizer.learning_rate * self.factor)


# In[5]:


import numpy as np
import tensorflow as tf
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)


# The model includes 4 different layers.  The first layer flatten is a input is a reshaping layer, this flattens the inputs.  The other three are Dense layers are dense layers, they are core layers that computes the transoformation between the 
# kernal and the input.  The arguments used are units, which determines the outputs(or nurons); our final layer needs to have 10 units (one per each number) 
# and dense included are two layers of relu and one layer of softmax.  

# In[6]:


model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])


# Here we compile the modle and calculate the crossentrophy loss.  Using a learning rate of.001 we are looking for the most accurate model.  

# In[7]:


model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(learning_rate=1e-3),
              metrics=["accuracy"])
expon_lr = ExponentialLearningRate(factor=1.005)


# Using the object for Exponential Learning (1.005) we can look at or fist model for analysis. 

# In[8]:


history = model.fit(X_train, y_train, epochs=1,
                    validation_data=(X_valid, y_valid),
                    callbacks=[expon_lr])


# As you can see from the model, the data seems to spike at about 1.  Hence the gradiance is greater than the loss.  

# In[ ]:


plt.plot(expon_lr.rates, expon_lr.losses)
plt.gca().set_xscale('log')
plt.hlines(min(expon_lr.losses), min(expon_lr.rates), max(expon_lr.rates))
plt.axis([min(expon_lr.rates), max(expon_lr.rates), 0, expon_lr.losses[0]])
plt.grid()
plt.xlabel("Learning rate")
plt.ylabel("Loss")


# Now we go through te process of creating and compiling the same model as above, however, we are looking at a learning rate of .1.  

# In[ ]:


keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)


# In[ ]:


model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])


# In[ ]:


model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(learning_rate=.1),
              metrics=["accuracy"])


# In[ ]:


import os
run_index = 1 # increment this at every run
run_logdir = os.path.join(os.curdir, "my_mnist_logs", "run_{:03d}".format(run_index))
run_logdir


# Finally, we create a model using 100 (I mean 30!) epochs. Because we set the patients to 20, we 'quit early' to avoid overfitting and unnecessary processing.  

# In[ ]:


early_stopping_cb = keras.callbacks.EarlyStopping(patience=20)
checkpoint_cb = keras.callbacks.ModelCheckpoint("my_mnist_model.h5", save_best_only=True)
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

history = model.fit(X_train, y_train, epochs=100,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_cb])


# # Model evaluations 
# It looks like we made our goal of 98%!  However, more impressibly the loss was only .06.  This is likely due to the relatively high learning rate that stayed below 1.  A low loss rate is in many cases a better way of demonstrating how a model is better (assuming it's not due to overfitting) than accuaracy.  

# In[ ]:


model = keras.models.load_model("my_mnist_model.h5") # rollback to best model
model.evaluate(X_test, y_test)


# In[ ]:




