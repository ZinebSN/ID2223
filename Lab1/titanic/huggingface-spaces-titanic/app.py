import gradio as gr
import numpy as np
from PIL import Image
import requests

import hopsworks
import joblib

project = hopsworks.login()
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("titanic_modal", version=5)
model_dir = model.download()
model = joblib.load(model_dir + "/titanic_model.pkl")


def titanic(sex, age, pclass, parch, embarked):
    input_list = []
    if pclass=='1 First Class':
        input_list.append(1)
    elif pclass=='2 Second Class':
        input_list.append(2)
    else:
        input_list.append(3)
   
    if sex=='Female':
        input_list.append(1)
    else:
        input_list.append(0)
    
    
    input_list.append(age)
    input_list.append(parch)
    
    if embarked=='C (Cherbourg)':
        
        input_list.append(2)
    elif embarked=='S (Southampton)':
        
        input_list.append(1)
    else:
        
        input_list.append(0)
   
    # 'res' is a list of predictions returned as the label.
    res = model.predict(np.asarray(input_list,dtype=object).reshape(1,-1)) 
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want 
    # the first element.
   
    
    flower_url = "https://raw.githubusercontent.com/avatar46/ID2223_lab1/main/images/" + str(res[0]) + ".png"
    img = Image.open(requests.get(flower_url, stream=True).raw)            
    return img
        
demo = gr.Interface(
    fn=titanic,
    title="Titanic Survival Predictive Analytics",
    description="Experiment with different entries to predict if the person will survive.",
    allow_flagging="never",
    ### Create user interface with 5 inputs
    inputs=[
        gr.inputs.Radio(default='Female', label="Gender", choices=['Female','Male']),
        gr.inputs.Slider(0,150,label='Age'),
        gr.inputs.Radio(default='1 First Class', label="Passenger Class ", choices=['1 First Class', '2 Second Class', '3 Third Class']),
        gr.inputs.Number(default=1.0, label="Parch: # of parents / children aboard the Titanic "),
        gr.inputs.Radio(default='C (Cherbourg)', label="Embarkation Port", choices=['C (Cherbourg)', 'Q (Queenstown)', 'S (Southampton)']),
        ],
    outputs=gr.Image(type="pil"))

demo.launch()

