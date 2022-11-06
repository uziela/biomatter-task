#!/usr/bin/env python

# Written by Karolis Uziela in 2022

import sys
import argparse
import os
from IPython import embed
import pandas as pd
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers
#import tensorflow_model_analysis as tfma
#from tensorflow.keras.layers import Dropout, Dense, BatchNormalization
from tensorflow.keras.regularizers import l2

from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
import datetime
import os
import json
import random

################################ Functions ####################################

def get_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    #parser.add_argument("input_file", help="Input file", type=str)
    parser.add_argument("input_csv", help="Input csv", type=str)
    parser.add_argument("log_dir", help="Tensorboard log dir. Will contain subdirectories named by datetime", type=str)
    parser.add_argument("output_dir", help="Will contain trained models, statistics, etc", type=str)
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")
    return parser.parse_args()


def write_file(output_file, out_str):
    """Write out_str to output_file"""
    with open(output_file, "w") as f:
        f.write(out_str)



def spearman(y_true, y_pred):
     return ( tf.py_function(spearmanr, [tf.cast(y_pred, tf.float32), 
                       tf.cast(y_true, tf.float32)], Tout = tf.float32) )


###################### Global constants and Variables #########################

# Global constants
# N/A

# Global variables
#LEARNING_RATE = 0.0001
#DROPOUT_RATE = 0.5
#L2_FACTOR = 0.00001
#
#N_HIDDEN1 = 600
#N_HIDDEN2 = 200

#N_TRAIN = 2000
N_TRAIN = 96

N_EPOCHS = 20000

N_RUNS = 100000

################################ Main script ##################################
    
if __name__ == '__main__':
    args = get_arguments()
    if args.verbose: 
        # When the script starts, print an information message with the name 
        # of the script and its  arguments
        sys.stderr.write("{} started running with arguments: {}\n".format(
            sys.argv[0], ' '.join(sys.argv[1:])))

    # Read data
    data = pd.read_csv(args.input_csv)
    labels = data.pop('score')

    train_data = data[:N_TRAIN]
    train_labels = labels[:N_TRAIN]

    val_data = data[N_TRAIN:]
    val_labels = labels[N_TRAIN:]

    N_VAL = val_labels.size

    # Create output dirs
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    for i in range(N_RUNS):
        print("Run iteration: {}".format(i))
        LEARNING_RATE = 0.002726 # 10**random.uniform(-3.5,-2.3)
        DROPOUT_RATE = 0.453436 #random.uniform(0.2,0.7)
        L2_FACTOR = 0.009610 # 10**random.uniform(-3,-1.5)
        N_HIDDEN1 = 679 # random.randint(400, 800)
        N_HIDDEN2 = 245 # random.randint(100, 400)
        date_string = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S") 
        params_string = "_LR_{}_DR_{}_L2_{}_NH1_{}_NH2_{}_EPOCHS_{}".format(LEARNING_RATE, DROPOUT_RATE, L2_FACTOR, N_HIDDEN1, N_HIDDEN2,N_EPOCHS)

        log_dir = os.path.join(args.log_dir, "logs_" + date_string + params_string)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        output_dir = os.path.join(args.output_dir, "output_" + date_string + params_string)
        print("Output dir: {}".format(output_dir))
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        #print(data.head())
        #print(labels.head())

        normalize = layers.Normalization()

        normalize.adapt(train_data)

        my_model = tf.keras.Sequential([
            normalize,
            layers.Dense(N_HIDDEN1, kernel_regularizer=l2(L2_FACTOR), activation="relu"),
            #layers.BatchNormalization(),
            layers.Dropout(DROPOUT_RATE),
            layers.Dense(N_HIDDEN2, kernel_regularizer=l2(L2_FACTOR), activation="relu"),
            #layers.BatchNormalization(),
            layers.Dropout(DROPOUT_RATE),
            layers.Dense(1)
        ])

        my_model.compile(loss = tf.keras.losses.MeanSquaredError(),
                        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                        metrics=spearman)
        #my_model.compile(loss = tf.keras.losses.MeanSquaredError(),
        #                optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, 'best_model'),
            save_weights_only=False,
            monitor='val_spearman',
            mode='max',
            save_best_only=True)

        my_model.fit(train_data, train_labels, epochs=N_EPOCHS, validation_data=(val_data, val_labels), batch_size=N_TRAIN, validation_batch_size = N_VAL, callbacks=[tensorboard_callback, model_checkpoint_callback], verbose=2)
        #my_model.fit(train_data, train_labels, epochs=N_EPOCHS, batch_size=N_TRAIN, callbacks=[tensorboard_callback], verbose=0)

        #results = my_model.evaluate(val_data, val_labels, batch_size=N_VAL)
        #print("Results:", results)

        train_preds = my_model.predict(train_data)
        val_preds = my_model.predict(val_data)
        
        stats = dict()
        stats["Training_MSE"] = mean_squared_error(train_labels, train_preds)
        stats["Validation_MSE"] = mean_squared_error(val_labels, val_preds)
        stats["Training_spearman_correlation"] = spearmanr(train_labels, train_preds)[0]
        stats["Validation_spearman_correlation"] = spearmanr(val_labels, val_preds)[0]
        
        print(stats)
        print("Saving model to: {}".format(os.path.join(output_dir, 'last_model')))

        my_model.save(os.path.join(output_dir, 'last_model'))
        with open(os.path.join(output_dir, 'statistics.json'), "w") as outfile:
            json.dump(stats, outfile, indent=4, sort_keys=True)

        out_str = "{},{},{},{},{},{},{},{},{},{}\n".format(LEARNING_RATE, DROPOUT_RATE, L2_FACTOR, N_HIDDEN1, N_HIDDEN2, N_EPOCHS, stats["Training_MSE"], stats["Validation_MSE"], stats["Training_spearman_correlation"], stats["Validation_spearman_correlation"])
        write_file(os.path.join(output_dir, 'statistics.csv'), out_str)

    if args.verbose:
        # Print another information message when the script finishes
        sys.stderr.write("{} done.\n".format(sys.argv[0]))



