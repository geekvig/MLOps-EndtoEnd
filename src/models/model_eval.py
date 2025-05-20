import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json

from dvclive import Live
import yaml
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import confusion_matrix
import mlflow


def load_data(filepath : str) -> pd.DataFrame:
    try:
         return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data from {filepath}:{e}")
#test_data = pd.read_csv("./data/processed/test_processed.csv")

# X_test = test_data.iloc[:,0:-1].values
# y_test= test_data.iloc[:,-1].values

def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame,pd.Series]:
    try:
        X = data.drop(columns=['Potability'],axis=1)
        y = data['Potability']
        return X,y
    except Exception as e:
        raise Exception(f"Error Preparing data:{e}")


def load_model(filepath:str):
    try:
        with open(filepath,"rb") as file:
            model= pickle.load(file)
        return model
    except Exception as e:
        raise Exception(f"Error loading model from {filepath}:{e}")
    
#model = pickle.load(open("model.pkl","rb"))
def evaluation_model(model, X_test:pd.DataFrame, y_test:pd.Series) -> dict:
    try:

        params = yaml.safe_load(open("params.yaml", "r"))
        test_size = params["data_collection"]["test_size"] 
        n_estimators = params["model_building"]["n_estimators"]
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test,y_pred)
        pre = precision_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        f1score = f1_score(y_test,y_pred)

        with mlflow.start_run():
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", pre)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1score)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("n_estimators", n_estimators)
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(5,5))
            sns.heatmap(cm, annot=True)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Confustion Matrix")
            plt.savefig("confusion_matrix.png")
            mlflow.log_artifact("confusion_matrix.png")
            mlflow.sklearn.log_model(model, "RandomForest")


        #Experiment tracking
        with Live(save_dvc_exp=True) as live:
            live.log_metric("accuracy", acc)
            live.log_metric("precision", pre)
            live.log_metric("recall", recall)
            live.log_metric("f1_score", f1score)
            live.log_param("test_size", test_size)
            live.log_param("n_estimators", n_estimators)

        metrics_dict = {

            'acc':acc,
            'precision':pre,
            'recall' : recall,
            'f1_score': f1score
        }
        return metrics_dict
    except Exception as e:
        raise Exception(f"Error evaluating model : {e}")


def save_metrics(metrics:dict,metrics_path:str) -> None:
    try:
        with open(metrics_path,'w') as file:
            json.dump(metrics,file,indent=4)
    except Exception as e:
        raise Exception(f"Error saving metrics to {metrics_path}:{e}")
    
def main():
    try:
        test_data_path = "./data/processed/test_processed.csv"
        model_path = "model.pkl"
        metrics_path = "metrics.json"

        test_data = load_data(test_data_path)
        X_test,y_test = prepare_data(test_data)
        model = load_model(model_path)
        metrics = evaluation_model(model,X_test,y_test)
        save_metrics(metrics,metrics_path)
    except Exception as e:
        raise Exception(f"An Error occurred:{e}")

if __name__ == "__main__":
    main()