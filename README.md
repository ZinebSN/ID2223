# ID2223

* Lab 1:

We used Modal, Hopsworks, and Huggingface to build a Serverless ML system for the [Titanic Dataset](https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv). Our work contains following steps.

1. Bulid a feature pipeline.
    * Data cleaning and Feature engineering on raw data (_titanic/titanic-feature-pipeline.py_): 
        * Choose 5 columns for model training: "Sex", "Age", "PClass", "Parch", "Embarked". 
        * Fill missing data by imputation using the mean for age column and transform categorical variables into numerical variables using map.
        * Write the features to Hopsworks as a Feature Group.

    * Writing synthetic data (_titanic/titanic-feature-pipeline-daily.py_): generate sythetic passenger data and store them as a Feature Group.

2. Write a training pipeline (_titanic/titanic-training-pipeline.py_): read training data from Hopsworks and use XGBoost classifier to train the model that predicts if a passenger survives or not.

3. Write a batch inference pipeline to predict if the synthetic passengers survived or not (_titanic/titanic-batch-inference-pipeline.py_).

4. Write Gradio applications and deploy them on Huggingface. The applications provides UI to allow users to predict passeger surivival with given features and to show prediction history.
    
    - Links to the HuggingFace spaces:
      - [Iris Predictive Analysis](https://huggingface.co/spaces/ZinebSN/iris)
      - [Iris Monitoring](https://huggingface.co/spaces/ZinebSN/iris-monitoring)
      - [Titanic Predictive Analysis](https://huggingface.co/spaces/ZinebSN/titanic)
      - [Titanic Monitoring](https://huggingface.co/spaces/ZinebSN/titanic-monitoring)
