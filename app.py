import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 


import datetime 
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import f1_score
import numpy as np
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense , Flatten , Conv1D,Conv2D,MaxPool1D
from tensorflow.keras.layers import MaxPool2D,Dropout,BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import BorderlineSMOTE,KMeansSMOTE

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense , Flatten , Conv1D,Conv2D,MaxPool1D
from tensorflow.keras.layers import MaxPool2D,Dropout,BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import RandomUnderSampler


import tensorflow.keras.backend as K
from imblearn.under_sampling import NearMiss

from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.utils import class_weight

from sklearn.naive_bayes import GaussianNB


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, zero_one_loss
from sklearn.metrics import classification_report



from sklearn.utils import resample

from sklearn.preprocessing import MinMaxScaler,StandardScaler



app = Flask(__name__)



def get_model() :
    model = Sequential()
    model.add(Conv1D(filters = 32, kernel_size = 3,padding = 'Same', activation ='relu', input_shape = (11,1)))
    model.add(Conv1D(filters = 32, kernel_size = 3,padding = 'Same', activation ='relu', ))
    model.add(MaxPool1D(2))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())


    model.add(Conv1D(filters = 64, kernel_size = 3,padding = 'Same', activation ='relu'))
    model.add(Conv1D(filters = 64, kernel_size = 3,padding = 'Same', activation ='relu'))
    model.add(MaxPool1D(2))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())

    model.add(Conv1D(filters = 128, kernel_size = 3,padding = 'Same', activation ='relu'))
    model.add(Conv1D(filters = 128, kernel_size = 3,padding = 'Same', activation ='relu'))
    model.add(MaxPool1D(2))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())
    
#     model.add(Conv1D(filters = 256, kernel_size = 3,padding = 'Same', activation ='relu'))
#     model.add(Conv1D(filters = 256, kernel_size = 3,padding = 'Same', activation ='relu'))
#     model.add(Dropout(0.25))
#     model.add(BatchNormalization())
    
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())
    model.add(Dense(2, activation = "softmax"))
    
    opt = keras.optimizers.Adam()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    model.summary()
    
    return model 







@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    features = [x for x in request.form.values()]
    

    if(features[1] == 'India') :  #'India'
        features[1] = 0 
    elif(features[1] == 'USA') : #USA 
        features[1] = 1 
    elif(features[1] == 'China') :  #China
        features[1] = 2 
    else :
        features[1] = 3 


    features[0] = float(features[0])//100
    
    if(features[2] == 'Male') :
        features[2] =  0
    elif(features[2]  == 'Female') :
        features[2] = 1
    else :
        features[2] = 2 

    features[3] = float(features[3])
    features[4] = float(features[4])
    features[5] = float(features[5])
    features[6] = float(features[6])
    features[7] = float(features[7])
    features[8] = float(features[8])
    features[9] = float(features[9])
    features[10] = float(features[10])



    model =  get_model()
    model.load_weights("my_model.ckpt")
    features = np.array(features)
    


    print("Features now ............ ",features,"\n")

    print("Done with converting features into np.array")
    features = np.asarray(features).astype(np.float32)



    print("Done with transformation and standarization    ",features,"\n")
    features = features.reshape(1,features.shape[0],1)

    

    print(features.shape,"   Done with reshaping")
    prediction = model.predict_classes(features)

    print("Done with predictions and it is ............" , prediction[0])

    return render_template('Output.html', prediction_text='Churn score predicted is .....  {}'.format(prediction[0]))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)