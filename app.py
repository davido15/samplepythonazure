from flask import Flask,request
from predict import predict_on_text
import json

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello, World!"
    
@app.route("/predict/<msg>")
def predict_toxicity(msg):
	return str(predict_on_text(msg))
    
