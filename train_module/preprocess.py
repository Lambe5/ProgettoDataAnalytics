from train import *
import pickle
def my_train_test(df_res):
    X = df_res.iloc[:, 1:]
    y = df_res[["Year"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return(X_train, X_test, y_train, y_test)

# def my_train_validation_test():
#     X = df_res.iloc[:, 1:]
#     y = df_res[["Year"]]

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
#     return(X_train, X_val, y_train, y_val, X_test, y_test)

#Normalization
# def norm(df, column_name, order):
#     x = df[column_name]
#     x_norm1 = np.linalg.norm(x, ord=order)
#     x_normalized = x / x_norm1
#     df[column_name] = x_normalized

#     if order == 1:
#         print(sum(x_normalized))
#     if order == 2:
#         print(sum(x_normalized**2))
#     if order == np.inf:
#         print(max(x_normalized))

# def min_max_sc(X_train,X_test,X_val = None):
#     #MinMax Scaling
#     min_max_scaler = preprocessing.MinMaxScaler()
#     min_max_scaler.fit(X_train)
#     X_train_minmax = min_max_scaler.transform(X_train)
#     X_test_minmax = min_max_scaler.transform(X_test)
#     #TODO: Salvare il file del modello con pickle
#     if X_val is not None:
#         X_val_minmax = min_max_scaler.transform(X_val)
#         return (X_train_minmax, X_test_minmax, X_val_minmax)
#     else:
#         return (X_train_minmax, X_test_minmax)

def standardization(X_train,X_test,X_val = None):
    #Standardization
    scaler = preprocessing.StandardScaler()
    #Train
    scaler.fit(X_train)
    #Application
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    #TODO: Salvare il file del modello con pickle
    file = open("scaler.save","wb")
    pickle.dump(scaler, file)
    file.close()
    # if X_val is not None:
    #     X_val_scaled = scaler.transform(X_val)
    #     return (X_train_scaled, X_test_scaled, X_val_scaled)
    # else:
    return (X_train_scaled, X_test_scaled)


def f_pca(X_train_scaled,X_test_scaled,num_components = None, X_val_scaled = None):
    if num_components is not None:
        pca = PCA(n_components=num_components)
    else: pca = PCA()
    principals_components_train = pca.fit_transform(X_train_scaled)

    # Trasforma il set di test utilizzando la stessa PCA addestrata sul set di addestramento
    principals_components_test = pca.transform(X_test_scaled)
    
    #TODO: Salvare il file del modello con pickle (?)
    file = open("PCA.save","wb")
    pickle.dump(pca)
    file.close()
    if X_val_scaled is not None:
        principals_components_val = pca.transform(X_val_scaled)
        return (principals_components_train, principals_components_test, principals_components_val, pca)
    else:
        return (principals_components_train, principals_components_test, pca)

