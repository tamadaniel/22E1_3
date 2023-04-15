"""
This is a boilerplate pipeline 'nome_modelo'
generated using Kedro 0.18.7
"""

import pyarrow.parquet as pq
import pandas as pd
import mlflow
from kedro_mlflow.io.metrics import MlflowMetricDataSet
from kedro_mlflow.io.artifacts import MlflowArtifactDataSet
from kedro_mlflow.io.models import MlflowModelSaverDataSet


tags = {
        "Projeto": "22E1_3_Kobe",
        "team": "Daniel S. U. Tamashiro",
        "dataset": "Kobe Bryant Shot Selection"
       }

def importar_dados(shot):
    
    mlflow.end_run()
    # absolute path to the CSV file
    file_path = "C:/Users/shtsu/Codigos/23E1_3/projeto/data/01_raw/kobe_dataset.csv"

    # read the CSV file into a pandas dataframe
    df = pd.read_csv(file_path)

    # filter column with 2PT Field Goal
    #filtered_df = df[df['shot_type'].str.contains('2PT Field Goal')]
    filtered_df = df[df['shot_type'].str.contains(shot)]

    # create a new dataframe with only the specified columns
    filtered_df = filtered_df.loc[:, ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance','shot_made_flag']]

    # drop rows with missing values
    filtered_df = filtered_df.dropna()

    # Track data loading with mlflow
    with mlflow.start_run(run_name="import_data"):
        # Log the file path and the number of rows loaded
        mlflow.log_param("file_path", file_path)
        mlflow.log_metric("size", len(filtered_df))
        #mlflow.log_metric("column", list(df.columns))

        # Save the filtered_df to MLflow
        mlflow.log_artifact("data/02_intermediate/filtered_df.parquet", "salvo filtered_df")

        # log the head of the dataframe as a parameter in MLflow
        with open("head.txt", "w") as f:
            f.write(filtered_df.head().to_string())
        mlflow.log_artifact("head.txt")
        #mlflow.log_param("head", filtered_df.head().to_string())

    return filtered_df

def data_metrics(data):
    mlflow.end_run()
    mlflow.start_run(run_name="data_metrics")
    metrics={
        'n_total': data.shape[0],
    }
    for klass, count in data['shot_made_flag'].value_counts().items():
        metrics[f'n_{klass}']=count

    

    return{
        key:{'value':value, 'step': 1}
        for key,value in metrics.items()
    }


def preparar_dados(filtered_df:pq.ParquetDataset,test_size,random_seed):

    from sklearn.model_selection import train_test_split
    import os

    mlflow.end_run()
    
    df=filtered_df

    # define o dataframe de entrada e a variável alvo
    X = df.drop('shot_made_flag', axis=1)
    y = df['shot_made_flag']

    # separa os dados em treino e teste com uma escolha aleatória e estratificada
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_seed)

    #reset index
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    y_train=y_train.to_frame()
    y_test=y_test.to_frame()


    # Track data loading with mlflow
    with mlflow.start_run(run_name="Preparacao_Dados"):
        # Log the file path and shape
        #mlflow.log_param("file_path", file_path)

        Xtrain_line,Xtrain_column=X_train.shape
        Xtest_line,Xtest_column=X_test.shape
        mlflow.log_metric("Xtrain_line", Xtrain_line)
        mlflow.log_metric("Xtrain_column", Xtrain_column)
        mlflow.log_metric("Xtest_line", Xtest_line)
        mlflow.log_metric("Xtest_column", Xtest_column)

        mlflow.log_metric("y_train shape", y_train.size)
        mlflow.log_metric("y_test shape", y_test.size)

        mlflow.log_metric("test size", test_size)
        mlflow.log_metric("random seed", random_seed)

        # Save the filtered_df to MLflow
        mlflow.log_artifact("C:/Users/shtsu/Codigos/23E1_3/projeto/data/03_primary/X_train.parquet", "X_train")
        mlflow.log_artifact("C:/Users/shtsu/Codigos/23E1_3/projeto/data/03_primary/X_test.parquet", "X_test")                    
        mlflow.log_artifact("C:/Users/shtsu/Codigos/23E1_3/projeto/data/03_primary/y_train.parquet", "y_train")
        mlflow.log_artifact("C:/Users/shtsu/Codigos/23E1_3/projeto/data/03_primary/y_test.parquet", "y_test")

        # log the head of the dataframe as a parameter in MLflow
        with open("X_train_head.txt", "w") as f:
            f.write(X_train.head().to_string())
        mlflow.log_artifact("X_train_head.txt")

        with open("X_test_head.txt", "w") as f:
            f.write(X_test.head().to_string())
        mlflow.log_artifact("X_test_head.txt")

        with open("y_train_head.txt", "w") as f:
            f.write(X_train.head().to_string())
        mlflow.log_artifact("y_train_head.txt")

        with open("y_test_head.txt", "w") as f:
            f.write(X_train.head().to_string())
        mlflow.log_artifact("y_test_head.txt")        


    return X_train, X_test, y_train, y_test


def model_lr(X_train:pq.ParquetDataset,X_test:pq.ParquetDataset,y_train:pq.ParquetDataset,y_test:pq.ParquetDataset):

    from sklearn.model_selection import train_test_split
    import pycaret.classification as pc
    from sklearn.metrics import log_loss
    import matplotlib.pyplot as plt

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    mlflow.end_run()

    data_train = pd.concat([X_train, y_train], axis=1)

    # Get the name of the target column (assumed to be the last column)
    target_col = data_train.columns[-1]

    clf = pc.setup(data = data_train, target = "shot_made_flag", verbose = False)
    lr = pc.create_model('lr')    
    tuned_pipe = pc.tune_model(lr)
    
    #pc.plot_model(tuned_pipe, plot='auc')
    #temp_name = "auc.png"
    #plt.savefig(temp_name)
    #mlflow.log_artifact(pc.plot_model(tuned_pipe, plot='auc'), "Curva ROC")

    y_pred = tuned_pipe.predict_proba(X_test)
    log_loss_score = log_loss(y_test, y_pred)
    #print('Log loss score: {:.4f}'.format(log_loss_score))

   # Track data loading with mlflow
    with mlflow.start_run(run_name="Treinamento_lr"):
        # Log pipeline parameters to MLflow
        mlflow.log_param("classifier", "LogisticRegression")
        
        # Track tags
        mlflow.set_tags(tags)

        # Track parameters
        mlflow.log_param("target", "shot_made_flag")
        mlflow.log_param("model", "lr")

        # Track metrics
        mlflow.log_metric("log_loss", log_loss_score)

        # Track model
        mlflow.sklearn.log_model(tuned_pipe, "model")
        
        #Registro do modelo
        #mlflow.log_artifact(local_path='./nodes.py',artifact_path='code')
        
    return clf


def report_model(clf):
    mlflow.end_run()
    mlflow.start_run(run_name="report model")

    metrics = clf.get_metrics()
        #convert the Pandas Series object to a dictionary
    metrics_dict=metrics.to_dict() 

    return metrics_dict

#function signature
def model_lda(shot, test_size, random_seed):

    from sklearn.model_selection import train_test_split
    import pycaret.classification as pc
    from sklearn.metrics import log_loss, f1_score
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    import numpy as np

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from mlflow.models.signature import infer_signature



    mlflow.end_run()


#####################################

   # absolute path to the CSV file
    file_path = "C:/Users/shtsu/Codigos/23E1_3/projeto/data/01_raw/kobe_dataset.csv"

    # read the CSV file into a pandas dataframe
    df = pd.read_csv(file_path)


    # filter column with 2PT Field Goal
    #filtered_df = df[df['shot_type'].str.contains('2PT Field Goal')]
    filtered_df = df[df['shot_type'].str.contains(shot)]

    # create a new dataframe with only the specified columns
    filtered_df = filtered_df.loc[:, ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance','shot_made_flag']]

    # drop rows with missing values
    filtered_df = filtered_df.dropna()

    
    df=filtered_df

    # define o dataframe de entrada e a variável alvo
    X = df.drop('shot_made_flag', axis=1)
    y = df['shot_made_flag']

    # separa os dados em treino e teste com uma escolha aleatória e estratificada
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_seed)

    #reset index
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    y_train=y_train.to_frame()
    y_test=y_test.to_frame()



###################################
    data_train = pd.concat([X_train, y_train], axis=1)

    # Get the name of the target column (assumed to be the last column)
    target_col = data_train.columns[-1]

    clf = pc.setup(data = data_train, target = "shot_made_flag", verbose = False)
    lda = pc.create_model('lda')    
    tuned_pipe = pc.tune_model(lda)
    
    y_pred = tuned_pipe.predict_proba(X_test)[:, 1]
  
    ######### choose best threshold
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    # calculate F1-score for each threshold
    f1_scores = []
    for threshold in thresholds:
        y_score = y_pred > threshold
        f1 = f1_score(y_test, y_score)
        f1_scores.append(f1)

    # find the index of the threshold that maximizes F1-score
    best_threshold_index = np.argmax(f1_scores)

    # get the best threshold and its corresponding F1-score
    best_threshold = thresholds[best_threshold_index]
    best_f1_score = f1_scores[best_threshold_index]

    y_pred_binary = np.where(y_pred >= best_threshold, 1, 0)
    log_loss_score = log_loss(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)
    #print('Log loss score: {:.4f}'.format(log_loss_score))

    #y_pred = tuned_pipe.predict_proba(X_test)

    # Define function signature
    signature = infer_signature(X_train, tuned_pipe.predict(X_train))

   # Track data loading with mlflow
    with mlflow.start_run(run_name="Treinamento_lda"):
        # Log pipeline parameters to MLflow
        mlflow.log_param("classifier", "Linear Discriminant Analysis (LDA)")
        
        # Track tags
        mlflow.set_tags(tags)

        # Track parameters
        mlflow.log_param("target", "shot_made_flag")
        mlflow.log_param("model", "lda")

        # Track metrics
        mlflow.log_metric("lda log_loss", log_loss_score)
        mlflow.log_metric("lda F1_score", f1)
        mlflow.log_metric("lda threshold", best_threshold)
        #mlflow.log_metric("lad F1_score", best_f1_score)

        # Track model
        mlflow.sklearn.log_model(tuned_pipe, "model", signature=signature)

    return tuned_pipe