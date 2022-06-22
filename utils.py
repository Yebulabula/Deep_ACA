#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_learning_curves(training_s_statistic, testing_s_statistic, testing_t_statistic):
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
        :param true_labels: ([string]): A list of ground-truth CPE-targets in testing data.
        :param predictions: ([string]): A list of CPE-targets predicted from the deep learning algorithms.
        :param activities: ([string]): The y-axis ticks of the confusion matrix.
    """
	CM = confusion_matrix(true_labels, predictions)
	plt.figure(figsize=(7, 5))
	sns.set(font_scale=1)
	sns.heatmap(CM, xticklabels=activities, yticklabels=activities,
                annot=True, fmt='d', cmap='Greens')
	plt.title('Confusion Matrix')
	plt.xlabel('Predicted Class')
	plt.ylabel('True Class')
	plt.savefig('/Users/yemao/Downloads/CDAN/CDAN_ACA/confusion_matrix.png')
	plt.show()

def save_log(obj, path):
	with open(path, 'wb') as f:
		pickle.dump(obj, f)
		print('[INFO] Object saved to {}'.format(path))

def save_model(model, path):
	torch.save(model.state_dict(), path)
	print("checkpoint saved in {}".format(path))

def load_model(model, path):
	"""
	Loads trained network params in case AlexNet params are not loaded.
	"""
	model.load_state_dict(torch.load(path))
	print("pre-trained model loaded from {}".format(path))
	
def _split_data_labels(data):
    """
        The simple function to split, shuffle and normalise data.
        :param data: (DataFrame): gblock or clinical isolate data frames.
        :return x: ([[int]]) A list of amplification curves, each of which is normalised 
        via the min-max scaler.
        :return y: ([string]) A list of CPE target names (i.e. labels in
        the training or testing set.). 
    """
    X, y = data.iloc[:,data.shape[1] - 45:], data[['Target']]
    X, y = shuffle(X, y, random_state = 2)
    X[X.columns] = MinMaxScaler().fit_transform(X)
    return X, y

def df_to_tensor(df):
    return torch.from_numpy(df.values).float().cpu()

def arr_to_tensor(arr):
    return torch.from_numpy(arr).float().cpu()

def _str_to_int(y):
	"""
        The function to convert CPE target names into integers. This function is
        used as the sparse-categorical entropy loss function is applied.
        i.e ('imp' -> 0, 'kpc' -> 1, 'ndm' -> 2, 'oxa48' -> 3, 'vim' -> 4)
        :param y: ([string]) The collection of CPE target names.
        :return: ([int]) A set of integers, each correponds to a CPE target name.
    """
	label_encoder = LabelEncoder()
	vec = label_encoder.fit_transform(y)
	return vec