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

#import tensorflow as tf
#from tensorflow.keras import layers
#import tensorflow_model_analysis as tfma
#from tensorflow.keras.layers import Dropout, Dense, BatchNormalization
#from tensorflow.keras.regularizers import l2

import lightgbm as lgb

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

N_TRAIN = 96

N_EPOCHS = 10000

N_RUNS = 1000000

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

    train_data_lgb = lgb.Dataset(train_data, label=train_labels)
    val_data_lgb = lgb.Dataset(val_data, label=val_labels)
    
    for i in range(N_RUNS):
        print("Run iteration: {}".format(i))
        min_data_in_leaf = random.randint(1, 50) 
        max_depth = random.randint(3, 7) 
        #num_leaves = 2**max_depth - 1 #random.randint(10, 3000)
        num_leaves = random.randint(10, 100)
        #max_bin = random.randint(200, 300) 
        lambda_l1 = random.randint(0, 100) 
        lambda_l2 = random.randint(0, 100) 
        learning_rate = 10**random.uniform(-2,-0.5)
        #min_gain_to_split = 0#random.randint(0, 15) 
        param = {'num_leaves': num_leaves, 'min_data_in_leaf': min_data_in_leaf, 'max_depth': max_depth, 'lambda_l1': lambda_l1, 'lambda_l2': lambda_l2, 'learning_rate': learning_rate}
        param['metric'] = 'mse'
        param['verbosity'] = -1
        param['feature_pre_filter'] = False
        print(param)

        date_string = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        params_string = "_numleaves_{}_mindata_{}_maxdepth_{}_lambdal1_{}_lambdal2_{}_learningrate_{}".format(num_leaves, min_data_in_leaf, max_depth, lambda_l1, lambda_l2, learning_rate)
        output_dir = os.path.join(args.output_dir, "lgb_output_" + params_string + "_" + date_string)

        bst = lgb.train(param, train_data_lgb, N_EPOCHS, valid_sets=[val_data_lgb], callbacks=[lgb.early_stopping(stopping_rounds=1000)])

        train_preds = bst.predict(train_data, num_iteration=bst.best_iteration)
        val_preds = bst.predict(val_data, num_iteration=bst.best_iteration)

        stats = dict()
        stats["Training_MSE"] = mean_squared_error(train_labels, train_preds)
        stats["Validation_MSE"] = mean_squared_error(val_labels, val_preds)
        stats["Training_spearman_correlation"] = spearmanr(train_labels, train_preds)[0]
        stats["Validation_spearman_correlation"] = spearmanr(val_labels, val_preds)[0]

        print(stats)

        # Create output dirs
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        out_str = "{},{},{},{},{},{},{},{},{},{}\n".format(
            num_leaves, min_data_in_leaf, max_depth, lambda_l1, lambda_l2, learning_rate, 
            stats["Training_MSE"], stats["Validation_MSE"], stats["Training_spearman_correlation"], stats["Validation_spearman_correlation"])
        write_file(os.path.join(output_dir, 'statistics.csv'), out_str)

    if args.verbose:
        # Print another information message when the script finishes
        sys.stderr.write("{} done.\n".format(sys.argv[0]))



