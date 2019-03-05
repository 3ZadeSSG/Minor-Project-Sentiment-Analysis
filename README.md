# Minor Project: Sentiment Analysis
This repository contains full code and deployment steps of my college's minor project on sentiment analysis.

Finally deployed model can be accessed from link: https://model-minor-project.herokuapp.com/ 

# Dataset used
I have used large movie reviews dataset from ai.stanford.edu

[**Link to dataset.**](http://ai.stanford.edu/~amaas/data/sentiment/)<br>

# Environment Used
Python 3.5.6 with everything in **requirements.txt**

**On Google Cloud (Used for training):**

         CPU: Intel(R) Xeon(R) CPU @ 2.20GHz
         GPU: Nvidia Tesla K80 12 GB
         RAM: 12 GB
         Storage: 358 GB 

**Local Machine (Used for deployment):**

        CPU: Intel(R) Core(TM) i7-7700K CPU @ 4.50GHz
        GPU: Nvidia GTX 1060 3GB
        RAM: 16 GB
        Storage: 240 GB SSD



# About the approach
The core of this project is Recurrent Neural Networks with LSTM cells. You can take a look at resources used to learn about LSTM cells from links below.
Running the whole network with LSTM cells defined in numpy will not be efficient from computation point of view, hence pytorch's implementation of LSTM has been used to train the network.
However a numpy implementation is defined in LSTMCell.py

[**Blog Post: Understanding LSTM Networks**](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)<br>
[**A detailed explanation of LSTM**](http://blog.echen.me/2017/05/30/exploring-lstms/)<br>
[**Youtube: LSTM Network implementation from scratch for generating texts.**](https://youtu.be/9zhrxE5PQgY)<br>

# Files required for network creation and training
1. **_myModelHelper.py :_** Used for processing the downloaded data and create the train, validation and testing datasets. 
2. **_Sentiment_Analysis_using_Stanford_AI_Dataset.ipynb :_** Main notebook containig all steps of network creating, training, testing, and checkpoint saving steps.

# Files required for deploying model on heroku cloud service.
1. **_SentimentAnalysisCheckpoint.pth :_** The cheackpoint file of trained model to be loaded so that we can use it for prediction
2. **_vocab_to_int.py :_** To avoid processing the data used in training to create the vocabulary to integer mapping dictonary, this file was written when running the main training program to avoid processing of data again, we can simply load this dictonary file to use in prediction.
3. **_commons.py :_** Python file containing the model definition, method to load the checkpoint and recreate model from that, and finally the prediction method which will return string containing prediction
4. **_app.py :_** Flask implementation for networking, it used GET and POST methods to receive a request and call prediction form *commons.py* and return result.
5. **_templates :_** Contains html files, each webpage of website to be rendered by calling from *app.py*
6. **_requirements.txt :_** All required files to be installed at Heroku Cloud to run the program, it contanis python version, numpy version, pytorch version etc.
7. **_runtime.txt :_** Required by Heroku for environment setup
8. **_app.json :_** Required by Heroku
9. **_Procfile :_** Required by Heroku Cloud to specify the type of app (web app in this case).


# Steps to deploy model on Heroku Cloud
1. (Optional) If case you want to use this repository to create your own app, clone this repo. with following command and goto step 3.
     
            $ git clone https://github.com/3ZadeSSG/Minor-Project-Sentiment-Analysis

2. After the training model, and saving the checkpoint, create another python file, (for example 'commons.py' in this case)
to recreate the model using checkpoint, and implement the predict function, so that passing any string will return result in string format.

3. Initialize the repo with (after removing any other git dependency from command line or by deleting git folder)
          
          $ git init

4. Add all files that are required by heroku (as mentioned above) using 

          $ git add <file_name>
   
   OR in case you have created another folder and copied all required files there, you can simply use
   
          $ git init
          $ git add *

5. Commit the files using

          $ git commit -m "<your commit message>" 
          
6. Create app using 

          $ heroku create <app_name>
          
7. Push changes to cloud, it will deploy the app, and give the website link to access that. 
        
          $ git push heroku master
    
**Note:** Heroku generally requires all files (including the ones which need to be downloaded using requirements.txt, means environment files) to be under 500 MB, if it goes over that, the request will be declined.
Also Heroku app runs on CPU, so it will take some time to process the result when we submit any input to app. 

================================================================
# Output

1. **When deploying the app**

<img src = "https://raw.githubusercontent.com/3ZadeSSG/Minor-Project-Sentiment-Analysis/master/Images/Heroku1.png">

2. **Message after successful deployment**

<img src = "https://raw.githubusercontent.com/3ZadeSSG/Minor-Project-Sentiment-Analysis/master/Images/Heroku2.png">

3. **Website with input window**

<img src = "https://raw.githubusercontent.com/3ZadeSSG/Minor-Project-Sentiment-Analysis/master/Images/Website Input.PNG">

4. **Giving input**

<img src = "https://raw.githubusercontent.com/3ZadeSSG/Minor-Project-Sentiment-Analysis/master/Images/Website Input 2.PNG">

5 **Prediction Result**

<img src = "https://raw.githubusercontent.com/3ZadeSSG/Minor-Project-Sentiment-Analysis/master/Images/Website Output.PNG">

 
# Running the prediction locally

To run the prediction locally, just import the **"commons.py"** and call the function **"getSentimentPredictionResult()"**
Here's a simple output of my terminal when i activated python.

        >>> from commons import *
        >>> sentence="kamikaze was one of the best album that came out last year"
        >>> print(getSentimentPredictionResult(sentence))
        0.1583  Positive sentence!
        
# Additional links of services used:
Each contains official links, and the main page has link for documentation
1. [**Flask**](http://flask.pocoo.org/)<br>
2. [**PyTorch**](https://pytorch.org/)<br>
3. [**Heroku**](https://www.heroku.com/)<br>



