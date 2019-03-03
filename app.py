from flask import Flask,render_template,request
import sys
import logging
app = Flask(__name__)
# commons.py contains all methods to load the model from checkpoint and predict the sentiment
from commons import *

@app.route("/",methods=['GET','POST'])
def hello():
	if request.method=='GET':
		# if GET method then render the homepage where data can be entered
		return render_template('index.html')
	if request.method=='POST':
		# if POST method then first call the prediction method to get result 
		# then render the result template and put result there
		file=request.form['file']
		predicted_result=getSentimentPredictionResult(file)
		return render_template('result.html',sentiment=predicted_result)
		
if __name__=='__main__':
	app.run(port=os.getenv('PORT',5000))
	app.logger.addHandler(logging.StreamHandler(sys.stdout))
	app.logger.setLevel(logging.ERROR)
