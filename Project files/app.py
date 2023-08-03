import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask,render_template,request

app=Flask(__name__)

model=load_model("breastcancerprediction.h5")

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method=='POST':
        f=request.files['image']
        basepath=os.path.dirname(__file__)
        filepath=os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)
        img=image.load_img(filepath,target_size=(180,180))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        pred=np.argmax(model.predict(x),axis=1)
        index=['BENIGN - bascially its a tumour no need to worry by performimg a simple surgery we can remove it ','MALIGNANT - its a serious condtion where cancer cells are spreading !! Hurry Up and perform operation']
        text="The patient is most likely a "  +str(index[pred[0]])
    return text
if __name__=='__main__':
    app.run(debug=False)