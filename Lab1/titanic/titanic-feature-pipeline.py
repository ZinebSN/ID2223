import os
import modal
import numpy as np
 
LOCAL=True

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4","joblib","seaborn","sklearn","dataframe-image"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()

def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()
    titanic_df = pd.read_csv("https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv")
    #titanic_df.info()
    #titanic_df.isna().sum
    titanic_df=titanic_df.drop(['Cabin','Ticket','PassengerId','Name','SibSp','Fare'], axis=1)
    titanic_df['Embarked']=titanic_df['Embarked'].replace(np.nan, 'Q')
    
    m = titanic_df['Age'].mean()
    #print(m)
    titanic_df['Age'] = titanic_df['Age'].replace(np.nan, int(m))
    sex_conversion = {
    'female': 1,  
    'male': 0}
    titanic_df['Sex'] = titanic_df['Sex'].map(sex_conversion)

    embarked_conversion = {
    'C': 2,
    'S': 1,  
    'Q': 0}
    titanic_df['Embarked'] = titanic_df['Embarked'].map(embarked_conversion)

    titanic_df.info()
    
    titanic_fg = fs.get_or_create_feature_group(
        name="titanic_modal",
        version=1,
        primary_key=["pclass","sex","age","embarked","parch"], 
        description="Titanic dataset")
    titanic_fg.insert(titanic_df, write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()
