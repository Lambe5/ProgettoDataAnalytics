MY_UNIQUE_NAME = "Lambertini-Marzocchi"

def getName():
    return MY_UNIQUE_NAME

import pandas as pd
import numpy as np
import seaborn as sns
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



def load_dataset(FILENAME):
    return pd.read_csv(FILENAME)
    

def my_train_test(df_res):
    X = df_res.iloc[:, 1:]
    y = df_res[["Year"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return(X_train, X_test, y_train, y_test)

def my_train_validation_test():
    X = df_res.iloc[:, 1:]
    y = df_res[["Year"]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    return(X_train, X_val, y_train, y_val, X_test, y_test)

#Normalization
def norm(df, column_name, order):
    x = df[column_name]
    x_norm1 = np.linalg.norm(x, ord=order)
    x_normalized = x / x_norm1
    df[column_name] = x_normalized

    if order == 1:
        print(sum(x_normalized))
    if order == 2:
        print(sum(x_normalized**2))
    if order == np.inf:
        print(max(x_normalized))

def min_max_sc(X_train,X_test,X_val = None):
    #MinMax Scaling
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(X_train)
    X_train_minmax = min_max_scaler.transform(X_train)
    X_test_minmax = min_max_scaler.transform(X_test)
    #TODO: Salvare il file del modello con pickle
    if X_val is not None:
        X_val_minmax = min_max_scaler.transform(X_val)
        return (X_train_minmax, X_test_minmax, X_val_minmax)
    else:
        return (X_train_minmax, X_test_minmax)

def standardization(X_train,X_test,X_val = None):
    #Standardization
    scaler = preprocessing.StandardScaler()
    #Train
    scaler.fit(X_train)
    #Application
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    #TODO: Salvare il file del modello con pickle
    
    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)
        return (X_train_scaled, X_test_scaled, X_val_scaled)
    else:
        return (X_train_scaled, X_test_scaled)


def f_pca(X_train_scaled,X_test_scaled,num_components = None, X_val_scaled = None):
    if num_components is not None:
        pca = PCA(n_components=num_components)
    else: pca = PCA()
    principals_components_train = pca.fit_transform(X_train_scaled)

    # Trasforma il set di test utilizzando la stessa PCA addestrata sul set di addestramento
    principals_components_test = pca.transform(X_test_scaled)
    #TODO: Salvare il file del modello con pickle (?)

    if X_val_scaled is not None:
        principals_components_val = pca.transform(X_val_scaled)
        return (principals_components_train, principals_components_test, principals_components_val, pca)
    else:
        return (principals_components_train, principals_components_test, pca)

def preprocess(df_file_name):
    df_res = load_dataset(df_file_name)

    X_train, X_test, y_train, y_test = my_train_test(df_res)
    # dato che non dovremo testare il modello ha senso fare anche il validation (?)
    
    X_train_minmax, X_test_minmax = min_max_sc(X_train=X_train,X_test=X_test) #ci serve a qualcosa?
    X_train_scaled, X_test_scaled = standardization(X_train,X_test)
    
    #PCA
    principals_components_train, principals_components_test, pca = f_pca( X_train_scaled=X_train_scaled,
                                                                          X_test_scaled=X_test_scaled,
                                                                            num_components=54)
    return principals_components_train, principals_components_test

#####################~~~MODELING~~~#####################

def modeling(X_train, y_train, X_test, y_test):
    return ""

def linearReg(X_train, y_train, X_test, y_test):
    #Linear-Regressor
    reg = LinearRegression().fit(X_train, y_train)
    #TODO: Salvare il file del modello con pickle
    predictions = reg.predict(X_test)

    metrics = calc_metrics(y_test,predictions,"linear_Regression")
    return metrics


def randomForest(X_train, y_train, X_test, y_test, n_alberi):
    rf_regressor = RandomForestRegressor(n_estimators=400,#TODO: da verificare iper parametri
                                                  max_depth= 140,
                                                  min_samples_split= 15,
                                                   random_state=42, n_jobs = -1)
    rf_regressor.fit(X_train, y_train)
    #TODO: Salvare il file del modello con pickle
    rf_predictions = rf_regressor.predict(X_test)
    return calc_metrics(y_test,rf_predictions,"RandomForest")



def svr(num_folds, X_train, y_train, X_test, y_test):
    # Definisci la griglia dei parametri da testare
    param_grid = {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': [1, 10, 50, 100]
    }
    svm_regressor = SVR()
    # initializing KFold
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    grid_search = GridSearchCV(estimator=svm_regressor, param_grid=param_grid, cv=kf, scoring='neg_mean_squared_error')

    # cross-validation to find best parameters
    grid_search.fit(X_train, y_train)
    best_svm_regressor = grid_search.best_estimator_

    # evaluate on best estimator
    y_pred = best_svm_regressor.predict(X_test)
    return calc_metrics(y_test, y_pred,"SVR")
    
def knn_cv(X_train, y_train, X_test, y_test):
    
    param_grid = {'n_neighbors': np.arange(2, 21)}

    knn = KNeighborsRegressor(n_jobs=-1)

    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
    grid_search.fit(X_train, y_train)

    # Inizializza la ricerca grid per MAE
    grid_search_mae = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_absolute_error', return_train_score=True)
    grid_search_mae.fit(X_train, y_train)

    # Best model for estimator
    best_model = grid_search.best_estimator_

    # predict best model on test set
    y_pred_test = best_model.predict(X_test)

    return calc_metrics(y_pred_test,y_test)

### FEED FORWARD
class FeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class FeedForwardPlus(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, depth=1):
        super(FeedForwardPlus, self).__init__()
        
        model = [
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        ]

        block = [
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        ]

        for i in range(depth):
            model += block
            print("i = ", i)

        
        self.model = nn.Sequential(*model)
        
        self.output = nn.Linear(hidden_size, num_classes)
        

    def forward(self, x):
        h = self.model(x)
        out = self.output(h)
        return out

def prepare_data(X_train, y_train, X_val, y_val):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)

    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)

    return train_dataset, val_dataset

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            #loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def evaluate_model(model, val_loader):
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            predictions.extend(outputs.numpy())
            targets.extend(labels.numpy())
    predictions = np.array(predictions)
    targets = np.array(targets)

    return calc_metrics(targets,predictions, "Feed Forward")

def prepare_model(PCA_X_train,y_train,PCA_X_test,y_test):
    # Preparazione dei dati
    train_dataset, test_dataset = prepare_data(PCA_X_train, y_train, PCA_X_test, y_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader,test_loader

def loss_optimization_def(model):
    # Definizione della loss e dell'ottimizzatore
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return (criterion,optimizer)

def FF_Model(PCA_X_train, train_loader,test_loader):
    # Definizione del modello
    input_size = PCA_X_train.shape[1]
    hidden_size = 200
    num_classes = 1
    model = FeedForward(input_size, hidden_size, num_classes)

    (criterion,optimizer) = loss_optimization_def(model)

    # Addestramento del modello
    num_epochs = 10
    train_model(model, train_loader, criterion, optimizer, num_epochs)

    # Valutazione del modello
    return evaluate_model(model, test_loader)

def FFPlus_model(PCA_X_train, train_loader,test_loader):
    # Definizione del modello
    input_size = PCA_X_train.shape[1]
    hidden_size = 800
    num_classes = 1
    depth = 1

    model = FeedForwardPlus(input_size, hidden_size, num_classes, depth)

    # Definizione della loss e dell'ottimizzatore
    criterion ,optimizer = loss_optimization_def(model)

    # Addestramento del modello
    num_epochs = 7
    train_model(model, train_loader, criterion, optimizer, num_epochs)

    # Valutazione del modello
    return (evaluate_model(model, test_loader))

#### 

#### TAB-NET ####
def TabNet_regression(
    regression_data,
    multi_target,
    continuous_cols,
    categorical_cols,
    continuous_feature_transform,
    normalize_continuous_features,
    target_range,
    batch_size=400,
    epochs = 7
):
    (train, test, target) = regression_data
    train_data, val_data = train_test_split(train, test_size=0.2, random_state=42)
    data_config = DataConfig(
        target=target + ["MedInc"] if multi_target else target,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        continuous_feature_transform=continuous_feature_transform,
        normalize_continuous_features=normalize_continuous_features,
        #num_workers=num_workers
    )
    model_config_params = {"task": "regression", "metrics":["mean_absolute_percentage_error","mean_absolute_error", "r2_score"]}
    if target_range:
        _target_range = []
        for target in data_config.target:
            _target_range.append(
                (
                    float(train_data[target].min()),
                    float(train_data[target].max()),
                )
            )
        model_config_params["target_range"] = _target_range
    model_config = TabNetModelConfig(**model_config_params)
    trainer_config = TrainerConfig(
        max_epochs=epochs,
        checkpoints=None,
        early_stopping=None,
        accelerator="cpu",
        fast_dev_run=False,
        batch_size= batch_size
    )
    optimizer_config = OptimizerConfig()
 
    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )
    tabular_model.fit(train=train_data, validation=val_data)
    #TODO: Salvare il modello
    result = tabular_model.evaluate(test)
    
    pred_df = tabular_model.predict(test)
    assert pred_df.shape[0] == test.shape[0]
    result["model"] = "Tabnet"
    return result
    
def input_Tab_tests(PCA_train, PCA_test):
    ref_df_train = pd.DataFrame(PCA_train)
    ref_df_test = pd.DataFrame(PCA_test)
 
    ref_df_train.columns = ref_df_train.columns.astype(str)
    ref_df_test.columns = ref_df_test.columns.astype(str)
 
    lista = list(ref_df_train.columns)
 
    target_column = str(ref_df_train.columns[-1])
   
    return (ref_df_train, ref_df_test, lista,  target_column)


def TabNet_model(PCA_train,PCA_test):
    (ref_df_train, ref_df_test, lista,  target_column) = input_Tab_tests(PCA_train,PCA_test)

    return TabNet_regression(regression_data=(ref_df_train, ref_df_test, [target_column]), multi_target=False,
        continuous_cols=lista,
        categorical_cols=[],
        continuous_feature_transform=None,
        normalize_continuous_features=True,
        target_range=False
    )


####

#### TAB-TANSFORMER ####
def tabtranformer_regression(
    regression_data,
    multi_target,
    continuous_cols,
    categorical_cols,
    continuous_feature_transform,
    normalize_continuous_features,
    target_range,
    epoch = 10,
    batch_size = 500
):
    (train, test, target) = regression_data
    data_config = DataConfig(
        target=target + ["MedInc"] if multi_target else target,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        continuous_feature_transform=continuous_feature_transform,
        normalize_continuous_features=normalize_continuous_features
    )
    model_config_params = {
        "task": "regression",
        "input_embed_dim": 8,
        "num_attn_blocks": 1,
        "num_heads": 2,
        "metrics":["mean_absolute_percentage_error","mean_absolute_error","r2_score"]
    }
    if target_range:
        _target_range = []
        for target in data_config.target:
            _target_range.append(
                (
                    float(train[target].min()),
                    float(train[target].max()),
                )
            )
        model_config_params["target_range"] = _target_range
    model_config = TabTransformerConfig(**model_config_params)
    trainer_config = TrainerConfig(
        max_epochs= epoch,
        checkpoints=None,
        early_stopping=None,
        accelerator="cpu",
        fast_dev_run=False,
        batch_size= batch_size
    )
    optimizer_config = OptimizerConfig()

    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )
    #print(train[53])
    tabular_model.fit(train=train)

    result = tabular_model.evaluate(test)
    #assert "test_mean_squared_error" in result[0].keys()
    pred_df = tabular_model.predict(test)
    assert pred_df.shape[0] == test.shape[0]
    result["model"]= "TabTransformer"
    return result

def TabNet_model(PCA_train,PCA_test):
    (ref_df_train, ref_df_test, lista,  target_column) = input_Tab_tests(PCA_train,PCA_test)
    return tabtranformer_regression(regression_data=(ref_df_train, ref_df_test, [target_column]), multi_target = None,
        continuous_cols = lista,
        categorical_cols = [],
        continuous_feature_transform = None,
        normalize_continuous_features = False,
        target_range=True, batch_size= 500, epoch=1)
    
####


def calc_metrics(y_true,y_pred,model="n/s"):
    return {
            "mae_": mean_absolute_error(y_true,y_pred),
            "mse_": mean_squared_error(y_true, y_pred),
            "mape_": mean_absolute_percentage_error(y_true, y_pred),
            "r2score_": r2_score(y_true, y_pred),
            "model": model
        }

