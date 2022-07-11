import pickle
import torch
import torch.nn.init as init
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_learning_curves(training_s_statistic, testing_s_statistic, testing_t_statistic):
    """
        This method is to plot leraning curves using training info.

        INPUT:
        training_s_statistic: statistic info on training soure domain data.
        testing_s_statistic: statistic info on testing soure domain data.
        testing_t_statistic: statistic info on training target domain data.
    """
    sns.set_theme()
    fig, axes = plt.subplots(1, 2, figsize=(10,5))
    y1 = [i[0]['classification_loss']  for i in training_s_statistic]
    y2 = [i[0]['cdan_loss'] for i in training_s_statistic]

    df = pd.DataFrame({"classification_loss": y1,
				"cdan_loss (Transfer Loss)" :y2})

    sns.lineplot(data = df, ax= axes[0])
    axes[0].set_xlabel("Number of Epochs", fontsize = 10)
    axes[0].set_ylabel("Loss", fontsize = 10)
    axes[0].set_title("Loss vs. Number of Epochs")
    
    y3 = [i['accuracy %'].item()  for i in testing_s_statistic]
    y4 = [i['accuracy %'].item() for i in testing_t_statistic]
    
    df1 = pd.DataFrame({"G-block Acc.": y3,
					   "Clinical Acc.":y4})
    sns.lineplot(data = df1, ax= axes[1])
    axes[1].set_xlabel("Number of Epochs", fontsize = 10)
    axes[1].set_ylabel("Accuracy", fontsize = 10)
    axes[1].set_title("Accuracy vs. Number of Epochs")
    plt.show()

def plot_cm(true_labels, predictions, activities):
	"""
        The function to plot confusion matrix using true and predicted CPE-targets. 
        
        INPUT:
        true_labels: ground truth labels in target domain.
        predictions: predicted labels.
        activities: the categories of CPE-targets.
    """
	CM = confusion_matrix(true_labels, predictions)
	plt.figure(figsize=(7, 5))
	sns.set(font_scale=1)
	sns.heatmap(CM, xticklabels=activities, yticklabels=activities,
                annot=True, fmt='d', cmap='Greens')
	plt.title('Confusion Matrix')
	plt.xlabel('Predicted Class')
	plt.ylabel('True Class')
	plt.savefig('CDAN/CDAN_ACA/confusion_matrix.png')
	plt.show()

def save_log(obj, path):
    """
        Save training log info to the specified path.
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
        print('[INFO] Object saved to {}'.format(path))

def save_model(model, path):
    """
        Save trained network params to the specified path.
    """
    torch.save(model.state_dict(), path)
    print("checkpoint saved in {}".format(path))

def load_model(model, path):
	"""
	    Loads trained network params in case Transfromer params are not loaded.
	"""
	model.load_state_dict(torch.load(path))
	print("pre-trained model loaded from {}".format(path))
	
def split_data_labels(data):
    """
        The simple function to split, shuffle and normalise data.
        
        INPUT:
        data: gblock or clinical isolate data frames.
        
        RETURN:
        x: A list of amplification curves, each of which is normalised via the min-max scaler.
        y: A list of CPE target names (i.e. labels in the training or testing set.). 
    """
    X, y = data.iloc[:,data.shape[1] - 45:], data[['Target']]
    X, y = shuffle(X, y, random_state = 2)
    X[X.columns] = MinMaxScaler().fit_transform(X)
    return X, y

def df_to_tensor(df):
    """
        The function to convert a dataframe to a tensor.
    """
    return torch.from_numpy(df.values).float().cpu()

def arr_to_tensor(arr):
    """
        The function to convert a np.array to a tensor.
    """
    return torch.from_numpy(arr).float().cpu()

def str_to_int(y):
	"""
        The function to convert CPE target names into integers. This function is
        used as the sparse-categorical entropy loss function is applied.
        i.e ('imp' -> 0, 'kpc' -> 1, 'ndm' -> 2)

        INPUT:
        y: The collection of CPE target names.

        RETURN:
        A set of integers, each correponds to a CPE target name.
    """
	label_encoder = LabelEncoder()
	vec = label_encoder.fit_transform(y)
	return vec

def sigmoid5_un(x, Fm, Fb, Sc, Cs, As):
	"""
        Five paramter model (universal notations).
        
        INPUT:
        x: iterative x locations
        Fm, Fb, Sc, Cs, As: parameters
        
        Fm: maximum fluorescence
        Fb: background fluorescence
        Sc: slope of the curve 
        Cs: fractional cycle of the inflection point (1/c)
        As: asymmetric shape (Richard's coefficient)
        
        RETURN:
        y outputs
    """
	return Fm / (1. + np.exp(-(x-Cs)*Sc))**As + Fb

def preprocess_data_dl(X,y):
    """
        The simple function to split, shuffle and normalise data.
        :param data: (DataFrame): gblock or clinical isolate data frames.
        :return x: ([[int]]) A list of amplification curves, each of which is normalised 
        via the min-max scaler.
        :return y: ([string]) A list of CPE target names (i.e. labels in
        the training or testing set.). 
    """
    X, y = shuffle(X, y, random_state = 2)
    for i in range(X.shape[-1]):
        X[:,:, i] = MinMaxScaler().fit_transform(X[:,:,i])
    return X, y