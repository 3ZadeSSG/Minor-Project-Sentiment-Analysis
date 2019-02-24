from flask import Flask,render_template,request
app = Flask(__name__)
from commons import *

@app.route("/",methods=['GET','POST'])
def hello():
	if request.method=='GET':
		return render_template('index.html')
	if request.method=='POST':
		file=request.form['file']
		#print("\n\t\tRequest RECEIVED")
		predicted_result=getSentimentPredictionResult(file)
		return render_template('result.html',sentiment=predicted_result)
		
if __name__=='__main__':
	app.run(port=os.getenv('PORT',5000))


