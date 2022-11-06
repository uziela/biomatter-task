#!/usr/bin/env python

# Written by Karolis Uziela in 2022

from genericpath import isdir
import sys
import argparse
import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sn

from IPython import embed

################################ Functions ####################################

def get_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    #parser.add_argument("input_file", help="Input file", type=str)
    parser.add_argument("input_csv", help="Input csv", type=str)
    parser.add_argument("output_dir", help="Output dir for plots", type=str)
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")
    return parser.parse_args()

def read_file(input_file):
    """Read file line by line and do... something"""
    with open(input_file) as f:
        while True:
            line = f.readline()
            if len(line) == 0:
                break
            line = line.rstrip('\n')
            #print(line)
            #bits = line.split("\t")


def write_file(output_file, out_str):
    """Write out_str to output_file"""
    with open(output_file, "w") as f:
        f.write(out_str)

def list_directory(input_dir, ends_with=""):
    """ List files in a directory 
    Arguments:
        input_dir -- directory to read
        ends_with -- list only files whose filename end match "ends_with" 
                     pattern (optional)
    """
    files = []
    for f in os.listdir(input_dir):
        if os.path.isfile(os.path.join(input_dir, f)) and \
                          f.endswith(ends_with):
            files.append(os.path.join(input_dir, f))
    return files

def plot_df(mydf, out_file):
    plt.figure()
    mydf.boxplot(rot=90)
    plt.subplots_adjust(bottom=0.4)
    plt.savefig(os.path.join(args.output_dir, out_file))
    print("--------------")
    print("Summary of {}".format(out_file))
    print(mydf.describe())

###################### Global constants and Variables #########################

# Global constants
# N/A

# Global variables
# N/A

################################ Main script ##################################
    
if __name__ == '__main__':
    args = get_arguments()
    if args.verbose: 
        # When the script starts, print an information message with the name 
        # of the script and its  arguments
        sys.stderr.write("{} started running with arguments: {}\n".format(
            sys.argv[0], ' '.join(sys.argv[1:])))

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    df = pd.read_csv(args.input_csv)
    df_norm = df / df.abs().max()

    df_train = df[:96]
    df_train_norm = df_train / df_train.abs().max()

    df_val = df[96:]
    df_val_norm = df_val / df_val.abs().max()

    #plt.margins(5)
    plot_df(df, "df.pdf")
    plot_df(df_norm, "df_norm.pdf")
    plot_df(df_train, "df_train.pdf")    
    plot_df(df_train_norm, "df_train_norm.pdf")
    plot_df(df_val, "df_val.pdf")    
    plot_df(df_val_norm, "df_val_norm.pdf")

    corr_matrix = df.corr().abs()
    plt.figure(figsize=(30, 30))
    sn.heatmap(corr_matrix, annot=True, vmin=0, vmax=1)
    plt.savefig(os.path.join(args.output_dir, "correlation_matrix_heatmap.pdf"))


    if args.verbose:
        # Print another information message when the script finishes
        sys.stderr.write("{} done.\n".format(sys.argv[0]))



