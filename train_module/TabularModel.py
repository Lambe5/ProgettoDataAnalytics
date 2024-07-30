from train_module.train import *
import pickle
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
    file = open("Tabnet.save","wb")
    pickle.dump(tabular_model,file)
    file.close()

    result = tabular_model.evaluate(test)
    
    pred_df = tabular_model.predict(test)
    assert pred_df.shape[0] == test.shape[0]
    result["model"] = "Tabnet"
    return result

def mergeX_Y(X,y):
    return pd.concat([X.reset_index(drop=True),
                      y.reset_index(drop=True)], axis=1)

def input_Tab_tests(principals_components_train,y_train,principals_components_test,y_test):
    ref_df_train = pd.DataFrame(principals_components_train)
    ref_df_test = pd.DataFrame(principals_components_test)
 
    ref_df_train.columns = ref_df_train.columns.astype(str)
    ref_df_test.columns = ref_df_test.columns.astype(str)
 
    lista = list(ref_df_train.columns)
    lista_target_range = list(range(1900, 2024))

    ref_df_train = mergeX_Y(ref_df_train,y_train)
    ref_df_test = mergeX_Y(ref_df_test,y_test)

    target_column = str(ref_df_train.columns[-1])
   
    return (ref_df_train, ref_df_test, lista, lista_target_range, target_column)


def TabNet_model(PCA_train,y_train,PCA_test,y_test):
    (ref_df_train, ref_df_test, lista,  target_column) = input_Tab_tests(PCA_train,y_train,PCA_test,y_test)

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

    # save the model 
    file = open("tabtransformer.save","wb")
    pickle.dump(tabular_model,file)
    file.close()

    result = tabular_model.evaluate(test)
    #assert "test_mean_squared_error" in result[0].keys()
    pred_df = tabular_model.predict(test)
    assert pred_df.shape[0] == test.shape[0]
    result["model"]= "TabTransformer"
    return result

def TabTransformer_model(PCA_train,y_train,PCA_test,y_test):
    (ref_df_train, ref_df_test, lista,  target_column) = input_Tab_tests(PCA_train,y_train,PCA_test,y_test)
    return tabtranformer_regression(regression_data=(ref_df_train, ref_df_test, [target_column]), multi_target = None,
        continuous_cols = lista,
        categorical_cols = [],
        continuous_feature_transform = None,
        normalize_continuous_features = False,
        target_range=True, batch_size= 500, epoch=1)
    
####