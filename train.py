import pickle
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
import nltk
import re

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from utils import *	
from process import *
warnings.filterwarnings("ignore")
import nltk

model_dir = "./model_new3"
tfidf_file ="tfidf_vect.pkl"
features_file = "feature_columns.json"
model_file = "model.pkl"
data_filename = "./raw_data.csv"

if not os.path.exists(model_dir):
	os.mkdir(model_dir)

def build_model():
	print("Building model ..........")
	
	t = time.time()
	data = get_processed_dataset(data_filename, 6000, 5000)
	target = data["msg_label"]
	features = data.drop("msg_label",axis=1)

	del data

	X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=1)
	
	print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

	tfidf_vect = TfidfVectorizer(min_df=0.001)
	tf_fit = tfidf_vect.fit(X_train["clean_content"])

	with open(model_dir+"/"+tfidf_file,"wb") as fp:
		pickle.dump(tf_fit, fp)

	vocabulary = tfidf_vect.get_feature_names()

	X_train_features = tf_fit.transform(X_train["clean_content"])
	X_train_features = pd.DataFrame(X_train_features.toarray(),columns=vocabulary)
	X_train = pd.concat([X_train[["msg_length","per_punct","per_cap"]].reset_index(drop=True), X_train_features],axis=1)

	X_test_features = tf_fit.transform(X_test["clean_content"])
	X_test_features = pd.DataFrame(X_test_features.toarray(),columns=vocabulary)
	X_test = pd.concat([X_test[["msg_length","per_punct","per_cap"]].reset_index(drop=True), X_test_features],axis=1)

	print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


	sm = SMOTE(random_state = 33)
	X_train_new, y_train_new = sm.fit_sample(X_train, y_train)
	print(X_train_new.shape, y_train_new.shape, X_test.shape, y_test.shape)

	with open(model_dir+"/"+features_file,"w") as fp:
		json.dump(str(list(X_train_new.columns)),fp)

	xgb_clf = xgb.XGBClassifier(eta=0.3, max_depth=8)
	print("Started training ..........")
	xgb_clf.fit(X_train_new, y_train_new)
	with open(model_dir+"/"+model_file,"wb") as fp:
		pickle.dump(xgb_clf,fp)

	y_pred_logreg = xgb_clf.predict(X_test)
	get_model_statistics(y_test, y_pred_logreg)


if __name__== "__main__":
	build_model()