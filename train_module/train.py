MY_UNIQUE_NAME = "Lambertini-Marzocchi"

def getName():
    return MY_UNIQUE_NAME

import sys
sys.path.append("./")

import pandas as pd
import numpy as np
import seaborn as sns
import pickle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.categorical_encoders import CategoricalEmbeddingTransformer
from pytorch_tabular.models import TabNetModelConfig
from pytorch_tabular.models import TabTransformerConfig
#costum files
from preprocess import *
from svr import *
from linearReg import *
from calc_metrics import * 
from RandomForest import *
from kNN_CV import *
from FeedForward import *
from Tabnet import *
from train_module.TabularModel import *


#insert file name and source as given in example 
FILE_NAME = "../train.csv"






############# TRAIN PHASE #############
def load_dataset(FILENAME):
    return pd.read_csv(FILENAME)
    

def preprocess(df_file_name):
    df_res = load_dataset(df_file_name)

    X_train, X_test, y_train, y_test = my_train_test(df_res)
    # dato che non dovremo testare il modello ha senso fare anche il validation (?)
    
    X_train_scaled, X_test_scaled = standardization(X_train,X_test)
    
    #PCA
    principals_components_train, principals_components_test, pca = f_pca( X_train_scaled=X_train_scaled,
                                                                          X_test_scaled=X_test_scaled,
                                                                            num_components=54)
    return (principals_components_train, principals_components_test,y_train,y_test)

#####################~~~MODELING~~~#####################

def modeling(X_train, X_test,y_train, y_test):
    metrics = []
    # Linear Regression    
    LR_metrics = linearReg(X_train,y_train,X_test,y_test)
    metrics.append(LR_metrics)

    #Random Forest
    # RF_metrics = randomForest(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test)
    # metrics.append(RF_metrics)

    #SVR
    SVR_metrics = svr(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test, reduced_model = True)
    metrics.append(SVR_metrics)

    #KNN
    kNN_metrics = knn_cv(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test)
    metrics.append(kNN_metrics)

    #Feed Forward: not used for poor performance over FFplus
    # FF_metrics = FF_train(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test)
    # metrics.append(FF_metrics)

    # Feed Forward Plus
    # FFPlus_metrics = FFplus_train(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test)
    # metrics.append(FFPlus_metrics)

    # TabNet
    TabNet_metrics = TabNet_model(PCA_train=X_train,y_train=y_train,PCA_test=X_test,y_test=y_test)
    metrics.append(TabNet_metrics)
    
    # TabTransformer
    TabTransformer_metrics = TabTransformer_model(PCA_train=X_train,y_train=y_train,PCA_test=X_test,y_test=y_test)
    metrics.append(TabTransformer_metrics)


    return metrics





############

(PCA_train, PCA_test,y_train,y_test) = preprocess(FILE_NAME)

all_model_metrics = modeling(PCA_train,PCA_test,y_train,y_test)










