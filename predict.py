import pickle
import pandas as pd
from utils import *
from process import *
from collections import Counter
import argparse
import sys


model_dir = "./models"

tfidf_file ="tfidf_vect.pkl"
features_file = "feature_columns.json"
model_file = "model.pkl"
data_filename = "./raw_data.csv"


with open(model_dir+"/"+tfidf_file,"rb") as fp:
	tfidf_vect = pickle.load(fp)


with open(model_dir+"/"+features_file,"r") as fp:
	imp_features = json.load(fp)

with open(model_dir+"/"+model_file,"rb") as fp:
	model = pickle.load(fp)	

imp_features = imp_features.replace("[","").replace("]","").replace("'","").split(", ")

def process_message(row):
	content = row["msg_content"]
	row["per_punct"] = count_punct(content)
	row["per_cap"] = count_caps(content)
	row["msg_length"] = row["msg_length"] ** (1/7)
	row["per_punct"] = row["per_punct"] ** (1/4)
	row["per_cap"] = row["per_cap"] ** (1/6)
	clean_content = clean_text(content)
	row = row[["msg_length","per_punct","per_cap"]]
	X_count = count_vect.transform([clean_content])
	row = row.tolist()
	X_count = X_count.toarray()
	X_count = X_count[0].tolist()
	row.extend(X_count)
	return row



def predict_unknown_data():

	model = get_model()
	test = pd.read_csv("./testdataset.csv")

	count = 0
	for index, row in test.iterrows():
		features_df = process_message(row)
		fdict = dict()
		for idx in range(0,len(columns)):
			fdict[columns[idx]] = features_df[idx]
		imp_values = []
		for key,val in fdict.items():
			if key in imp_features:
				imp_values.append(val)
		imp_values = np.array(imp_values)
		y_pred = model.predict([imp_values])
		print(features_df[0],y_pred)
		if count == 100:
			break
		else:
			count += 1		

def process_text(content):
	per_punct = count_punct(content) ** (1/4)
	per_cap = count_caps(content) ** (1/6)
	msg_length = len(content) ** (1/7)
	clean_content = clean_text(content)
	row = [msg_length, per_punct, per_cap]
	X_count = tfidf_vect.transform([clean_content])
	X_count = X_count.toarray()
	X_count = X_count[0].tolist()
	row.extend(X_count)
	return row


def predict_on_text(msg):
	features_df = process_text(msg)
	fdict = pd.DataFrame(np.array(features_df).reshape(1,-1), columns = imp_features)
	y_pred = model.predict(fdict)
	print(y_pred)
	return y_pred[0]


	
if __name__ == "__main__":

	msg = sys.argv[1]
	flag = predict_on_text(msg)