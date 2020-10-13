import glob
import pandas as pd
import numpy as np
import json
import os
import sys
import time
import warnings
from sklearn.utils import shuffle
import string
from utils import *
import xgboost as xgb 
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

from itertools import islice

def take(n, iterable):
	"Return first n items of the iterable as a list"
	return list(islice(iterable, n))

def get_dataset(filename):
	data = pd.read_csv(filename)
	data = data.drop("Unnamed: 0",axis=1)
	data.rename(columns={'length': 'msg_length','content':'msg_content','label':'msg_label'}, inplace=True)
	data = shuffle(data).reset_index(drop=True)
	data = data[1000:]
	return data

def get_processed_dataset(filename, non_toxic_count, toxic_count):
	
	data = get_dataset(filename)

	# Pick comments with length less than the mean length
	data = data[data["msg_length"]<=data["msg_length"].mean()]

	# Get new features - percentage of capital letters and percentage of punctuation in a comment
	data["per_punct"] = data["msg_content"].apply(lambda x: count_punct(x))
	data["per_cap"] = data["msg_content"].apply(lambda x: count_caps(x))

	# Transform new features
	data["msg_length"] = data["msg_length"] ** (1/7)
	data["per_punct"] = data["per_punct"] ** (1/4)
	data["per_cap"] = data["per_cap"] ** (1/6)

	# Clean comments
	data["clean_content"] = data["msg_content"].apply(lambda s:clean_text(s))

	# Drop original comments
	data = data.drop("msg_content",axis=1)
	data["msg_label"] = data["msg_label"].astype(np.int64)

	# Pick non-toxic and toxic messages
	non_toxic = shuffle(data[data["msg_label"]==0]).reset_index(drop=True)[0:non_toxic_count]
	toxic = shuffle(data[data["msg_label"]==1]).reset_index(drop=True)[0:toxic_count]
	data = pd.concat([toxic, non_toxic])

	return data
