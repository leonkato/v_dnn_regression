#!/usr/bin/env python
# coding: utf-8

# ### 

# ---
# ---
# ---

##### This script needs ipython to run.
##### from command line do:  'ipthon v_script.py'
# 
# 
# * All four trained models are in the directory __'./models905'__.  Note that these are directories.  Models can be loaded with 
# ``` model = tf.keras.models.load_model('dir_path')```
#     * do_dnn_model
#     * nh4_dnn_model 
#     * no3_dnn_model  
#     * po4_dnn_model
# 
# * This directory contains the notebooks that build, train and save the models (using __'data905/training905.xlsx'__ for training data.
#     * model_builder_DO.ipynb  
#     * model_builder_NO3.ipynb
#     * model_builder_NH4.ipynb  
#     * model_builder_PO4.ipynb
# 
# * The directory __'./val_predictions'__ contains the predictions for each target.
#     * dnn_predicted_do.csv   
#     * dnn_predicted_no3.csv
#     * dnn_predicted_nh4.csv  
#     * dnn_predicted_po4.csv
#
# * The features come from __'./data905/validation905.xlsx'__

# ##### Make sure we have these modules

get_ipython().system('pip install -q pandas')
get_ipython().system('pip install -q sklearn')
get_ipython().system('pip install -q openpyxl')

##### Imports

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

print(f'tensor flow version is {tf.__version__}')

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

# ##### Prep GPU

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print(f'gpus ==> {gpus}')


# #### Some useful functions

def build_and_compile_model(normalizer):
  model = keras.Sequential([
      normalizer,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model

def plot_loss(history, var_name):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  # plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel(f'Error [{var_name}]')
  plt.legend()
  plt.grid(True)


# ---
# ---
# ---

# ##### Get the validation data and examine

val_data_path = "data905/validation905.xlsx"
print(f'Loading validation data: {val_data_path}')
val_data = pd.read_excel(val_data_path,na_values="NaN",  engine='openpyxl')
print('Validation data read..')

# val_data.head()

# for val in ['GHI', 'TEMP', 'PH',
#        'TURB', 'SA DO', 'SA NO3', 'SA NH4', 'SA PO4']:
#     val_data.plot(x='datetime', y=val, kind='scatter')


#################################################################
#  Beginning of DO
#################################################################


# ##### Predict DO

print('Beginning DO...')

model_path = 'models905/do_dnn_model'
print(f'Reloading model: {model_path}...')
reload_do_dnn_model = tf.keras.models.load_model('models905/do_dnn_model')
print('Done')
print()

print('Predicting DO...')
do_features = [ 'Month', 'Day', 'Week', 'GHI', 'TEMP', 'PH', 'EC', 'TURB']
do_val_yhats = reload_do_dnn_model.predict(val_data[do_features])
print('Done')


do_target = ['SA DO']
print('Checking DO metrics...')
print('r2 score: ', r2_score(val_data[do_target], do_val_yhats))
print('mean abs. error: ', mean_absolute_error(val_data[do_target], do_val_yhats))

# plt.scatter(val_data['datetime'], val_data[do_target], label='actual')
# plt.scatter(val_data['datetime'], do_val_yhats, color='red', alpha=0.5, label='pred')
# plt.xlabel('Actual DO')
# plt.ylabel('Predicted [DO]')
# plt.legend()
# plt.grid(True)
# plt.show()

# Save the predictions along with time and actual DO.

do_val_output = pd.DataFrame()
do_val_output['datetime'] = val_data['datetime']
do_val_output['predicted DO'] = do_val_yhats
do_val_output['actual DO'] = val_data['SA DO']
# print(do_val_output.head())

print('Showing plot actual vs. predicted')

do_val_output.plot(x='datetime', y=['predicted DO', 'actual DO'])
plt.show(block=False)

do_out_name = 'dnn_predicted_do.csv'
print(f'saving predictions in {do_out_name}...')
do_val_output.to_csv(do_out_name, index=False)
print('Done.')
print()

print('Target DO done.')
print()

#################################################################
#  End of DO
#################################################################


#################################################################
#  Beginning of NO3
#################################################################

print('Beginning no3...')

model_path = 'models905/no3_dnn_model'

print(f'Reloading model: {model_path}...')
reload_no3_dnn_model = tf.keras.models.load_model('models905/no3_dnn_model')
print('Done')
print()

print('Predicting no3...')
no3_features = [ 'Month', 'Day', 'Week', 'GHI', 'TEMP', 'PH', 'EC', 'TURB', 'SA DO']
no3_val_yhats = reload_no3_dnn_model.predict(val_data[no3_features])
print('Done')

no3_target = ['SA NO3']
print('Checking NO3 metrics...')
print('r2 score: ', r2_score(val_data[no3_target], no3_val_yhats))
print('mean abs. error: ', mean_absolute_error(val_data[no3_target], no3_val_yhats))

# plt.scatter(val_data['datetime'], val_data[no3_target], label='actual')
# plt.scatter(val_data['datetime'], no3_val_yhats, color='red', alpha=0.5, label='pred')
# plt.xlabel('Actual no3')
# plt.ylabel('Predicted [no3]')
# plt.legend()
# plt.grid(True)
# plt.show()

# Save the predictions along with time and actual no3.

no3_val_output = pd.DataFrame()
no3_val_output['datetime'] = val_data['datetime']
no3_val_output['predicted no3'] = no3_val_yhats
no3_val_output['actual no3'] = val_data['SA NO3']
# print(no3_val_output.head())

print('Showing plot actual vs. predicted')

no3_val_output.plot(x='datetime', y=['predicted no3', 'actual no3'])
plt.show(block=False)

# no3_val_output.to_csv('dnn_predicted_no3.csv', index=False)
no3_out_name = 'dnn_predicted_no3.csv'
print(f'saving predictions in {no3_out_name}...')
no3_val_output.to_csv(no3_out_name, index=False)
print('Done.')
print()


print('Target NO3 done.')
print()

#################################################################
#  End of NO3
#################################################################

#################################################################
#  Beginning of NH4
#################################################################

print('Beginning nh4...')
model_path = 'models905/nh4_dnn_model'

print(f'Reloading model: {model_path}...')
reload_nh4_dnn_model = tf.keras.models.load_model('models905/nh4_dnn_model')
print('Done')
print()

print('Predicting nh4...')
nh4_features = [ 'Month', 'Day', 'Week', 'GHI', 'TEMP', 
                'PH', 'EC', 'TURB', 'SA DO', 'SA NO3']
nh4_val_yhats = reload_nh4_dnn_model.predict(val_data[nh4_features])
print('Done')

nh4_target = ['SA NH4']
print('Checking NH4 metrics...')
print('r2 score: ', r2_score(val_data[nh4_target], nh4_val_yhats))

print('mean abs. error: ', mean_absolute_error(val_data[nh4_target], nh4_val_yhats))

# plt.scatter(val_data['datetime'], val_data[nh4_target], label='actual')
# plt.scatter(val_data['datetime'], nh4_val_yhats, color='red', alpha=0.5, label='pred')
# plt.xlabel('Actual nh4')
# plt.ylabel('Predicted [nh4]')
# plt.legend()
# plt.grid(True)
# plt.show()

# Save the predictions along with time and actual nh4.

nh4_val_output = pd.DataFrame()
nh4_val_output['datetime'] = val_data['datetime']
nh4_val_output['predicted nh4'] = nh4_val_yhats
nh4_val_output['actual nh4'] = val_data['SA NH4']
# print(nh4_val_output.head())

print('Showing plot actual vs. predicted')

nh4_val_output.plot(x='datetime', y=['predicted nh4', 'actual nh4'])
plt.show(block=False)

# nh4_val_output.to_csv('dnn_predicted_nh4.csv', index=False)
nh4_out_name = 'dnn_predicted_nh4.csv'
print(f'saving predictions in {nh4_out_name}...')
nh4_val_output.to_csv(nh4_out_name, index=False)
print('Done.')
print()


print('Target NH4 done.')
print()

#################################################################
#  End of NH4
#################################################################

#################################################################
#  Beginning of PO4
#################################################################

print('Beginning po4...')
model_path = 'models905/po4_dnn_model'

print(f'Reloading model: {model_path}...')
reload_po4_dnn_model = tf.keras.models.load_model('models905/po4_dnn_model')

print('Done')
print()

print('Predicting po4...')

po4_features = [ 'Month', 'Day', 'Week', 'GHI', 'TEMP', 
                'PH', 'EC', 'TURB', 'SA DO', 'SA NO3', 'SA NH4']
po4_val_yhats = reload_po4_dnn_model.predict(val_data[po4_features])

print('Done')

po4_target = ['SA PO4']
print('Checking PO4 metrics...')
print('r2 score: ', r2_score(val_data[po4_target], po4_val_yhats))

print('mean abs. error: ', mean_absolute_error(val_data[po4_target], po4_val_yhats))

# plt.scatter(val_data['datetime'], val_data[po4_target], label='actual')
# plt.scatter(val_data['datetime'], po4_val_yhats, color='red', alpha=0.5, label='pred')
# plt.xlabel('Actual po4')
# plt.ylabel('Predicted [po4]')
# plt.legend()
# plt.grid(True)
# plt.show()

# Save the predictions along with time and actual po4.

po4_val_output = pd.DataFrame()
po4_val_output['datetime'] = val_data['datetime']
po4_val_output['predicted po4'] = po4_val_yhats
po4_val_output['actual po4'] = val_data['SA PO4']
# print(po4_val_output.head())

print('Showing plot actual vs. predicted')

po4_val_output.plot(x='datetime', y=['predicted po4', 'actual po4'])
plt.show(block=False)

# po4_val_output.to_csv('dnn_predicted_po4.csv', index=False)
po4_out_name = 'dnn_predicted_po4.csv'
print(f'saving predictions in {po4_out_name}...')
po4_val_output.to_csv(po4_out_name, index=False)
print('Done.')
print()

print('Target PO4 done.')
print()

# this call to input leaves the charts up until you hit enter.

x = input('Press enter to dismiss charts.')
