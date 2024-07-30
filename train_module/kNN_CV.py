from train import *
import pickle

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
    
    # save the model
    file = open("kNN.save","wb")
    pickle.dump(best_model,file)
    file.close()
    
    # predict best model on test set
    y_pred_test = best_model.predict(X_test)

    return calc_metrics(y_pred_test,y_test, "kNN")
