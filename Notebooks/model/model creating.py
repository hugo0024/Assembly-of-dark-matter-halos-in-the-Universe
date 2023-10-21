#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Importing necessary libraries

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping



# In[ ]:


# Number of features in the data

# excluding 'pair' and 'will_merge' columns.

num_features = len(data.columns) - 2  


# In[ ]:


# Defining the model

model = Sequential()
model.add(Dense(128, input_dim=num_features, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))  # Output layer with sigmoid activation for binary classification


# In[ ]:


# Compiling the model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


# Summarising  the model

model.summary()


# In[ ]:


# Preparing the callbacks to save the best model and prevent overfitting.

checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)


# In[ ]:


# Preparing data for the model

X = data.drop(columns=['pair', 'will_merge']).values
y = data['will_merge'].values


# In[ ]:


# Splitting data into training and testing sets

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


# Fitting the model to the data

history = model.fit(
    X_train, 
    y_train, 
    epochs=100, 
    batch_size=32, 
    validation_data=(X_test, y_test),
    callbacks=[checkpoint, early_stopping]
)


# In[ ]:


# Evaluating the model

_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))

