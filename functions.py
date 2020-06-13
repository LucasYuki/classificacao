# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import os
from tensorflow import one_hot
from matplotlib import pyplot as plt

def verify_directory(dir_path):
    if os.path.isdir(dir_path):
        print(dir_path)
        print("directory already exist, please chose other name")
        input("Press enter to exit")  
        return False
    else:
        try:
            os.makedirs(dir_path)
        except:
            print("Creation of the directory %s failed\n" % (dir_path))
            input("Press enter to exit") 
            return False
    return True
    
def get_files(directory_df):
    files = pd.read_csv(directory_df, index_col=[0,1,2])
    columns = {}
    for i in range(len(files.index.names)):
        columns[files.index.names[i]] = list(files.index.unique(i))
    return (files, columns)

def read_data(path):
    return pd.read_csv(path, sep="	",  header=None, dtype="float")

def normalize(files):
    data = read_data(files.loc["data"][0]).to_numpy()
    mean = read_data(files.loc["mean"][0]).to_numpy()
    std  = read_data(files.loc["std"][0]).to_numpy()
    return (data-mean)/std

def get_all_data(grand, directory_df, norm=True):
    files, columns = get_files(directory_df)
    data = {}
    
    for cond in columns["Condicao"]:
        if grand=="Hfp" and cond=="disart":
            continue
        if norm:
            data[cond] = normalize(files.loc[grand, cond])
        else:
            data[cond] = read_data(files.loc[grand, cond].loc["data"][0]).to_numpy()
    return data

def get_splited_data(grand, train_len, val_len, test_len, directory_df, norm=True):
    files, columns = get_files(directory_df)
    Train = {"x":[], "y":[]}
    Val   = {"x":[], "y":[]}
    Test  = {"x":{}, "y":{}, "y one_hot":{}}
    
    count = 0
    for cond in columns["Condicao"]:
        if grand=="Hfp" and cond=="disart":
            continue
        if norm:
            temp = normalize(files.loc[grand, cond])
        else:
            temp = read_data(files.loc[grand, cond].loc["data"][0]).to_numpy()
        Train["x"].append(temp[:train_len, :])
        Train["y"].append(np.ones((train_len, 1), dtype=int)*count)
        Val["x"].append(temp[train_len:train_len+val_len, :])
        Val["y"].append(np.ones((val_len, 1), dtype=int)*count)
        Test["x"][cond] = temp[train_len+val_len:train_len+val_len+test_len, :]
        Test["y"][cond] = count
        count += 1

    Train["x"] = np.concatenate(Train["x"])
    Train["y"] = np.vstack(Train["y"])
    Val["x"] = np.concatenate(Val["x"])
    Val["y"] = np.vstack(Val["y"])
    for cond in Test["y"].keys():
        Test["y one_hot"][cond] = one_hot(np.ones((test_len,), dtype=int)*Test["y"][cond], 
                                    count, on_value=1, off_value=0)

    # embaralha os dados de treino
    index = np.random.permutation(Train["x"].shape[0])
    Train["x"] = Train["x"][index, :]
    Train["y"] = Train["y"][index].reshape((-1, Train["y"].shape[-1]))
    return Train, Val, Test, count
    
def predict(model, x, n_per_pred=100):
    predictions = []
    for i in range(n_per_pred):
        predictions.append(model(x))
    predictions = np.stack(predictions, axis=1)
    return predictions

def plot_train_hist(path, figsize=(10,5)):
    data = pd.read_csv(path+"\\train_history.csv")

    fig, ax = plt.subplots(figsize=figsize)  # Create a figure and an axes.
    ax.plot(data["epoch"], data["loss"], label='train', lw=0.5)  # Plot some data on the axes.
    ax.plot(data["epoch"], data["val_loss"], label='validate', lw=0.5)  # Plot more data on the axes...
    ax.set_xlabel('epoch')  # Add an x-label to the axes.
    ax.set_ylabel('loss')  # Add a y-label to the axes.
    ax.set_title("loss")  # Add a title to the axes.
    ax.legend()  # Add a legend.
    plt.grid(True)
    plt.savefig(path+"//train_history_loss.png", quality=100)
    plt.savefig(path+"//train_history_loss.SVG", quality=100)
    
    fig, ax = plt.subplots(figsize=figsize)  # Create a figure and an axes.
    ax.plot(data["epoch"], data["accuracy"], label='train', lw=0.5)  # Plot some data on the axes.
    ax.plot(data["epoch"], data["val_accuracy"], label='validate', lw=0.5)  # Plot more data on the axes...
    ax.set_xlabel('epoch')  # Add an x-label to the axes.
    ax.set_ylabel('accuracy')  # Add a y-label to the axes.
    ax.set_title("accuracy")  # Add a title to the axes.
    ax.legend()  # Add a legend.
    plt.grid(True)
    plt.savefig(path+"//train_history_accuracy.png", quality=100)
    plt.savefig(path+"//train_history_accuracy.SVG", quality=100)
    return [path+"//train_history_loss.png", path+"//train_history_accuracy.png"]

def heat_map(figsize,
             data,
             row_label,
             column_label,
             save_dir,
             name,
             title,
             vmax=1.00,
             format_text="%.2f"):
    
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(data, cmap="coolwarm", alpha=0.8, vmin=0.0, vmax=vmax)
    plt.colorbar(im, format=format_text)

    ax.set_xticks(np.arange(len(column_label)))
    ax.set_xticklabels(column_label)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="center")

    ax.set_yticks(np.arange(len(row_label)))
    ax.set_yticklabels(row_label)
    plt.ylim(len(row_label)-0.5, -0.5)


    for i in range(len(row_label)):
        for j in range(len(column_label)):
            ax.text(j, i, format_text % np.around(data[i][j], 2),
                    ha="center", va="center", color="black")

    ax.set_title(title, pad=10)
    fig.tight_layout()
    
    if not os.path.isdir(save_dir+"/figures PNG"):
        os.makedirs(save_dir+"/figures PNG")
    if not os.path.isdir(save_dir+"/figures SVG"):
        os.makedirs(save_dir+"/figures SVG")
    plt.savefig(save_dir + "/figures PNG/" + name + '.png', quality=100)
    plt.savefig(save_dir + "/figures SVG/" + name + '.SVG', quality=100)
    plt.close('all')