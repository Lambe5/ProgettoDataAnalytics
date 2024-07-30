from test import *
import pickle
def svr(num_folds, X_train, y_train, X_test, y_test, reduced_model):
    if reduced_model:
        X_train = X_train[:-191740,:]
        y_train = y_train.head(10000)
        X_test = X_test[:-47935]
        y_test.head(2500)

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
    
    # save model 
    file = open("SVR.save","wb")
    pickle.dump(best_svm_regressor,file)
    file.close()
    # evaluate on best estimator
    y_pred = best_svm_regressor.predict(X_test)
    return calc_metrics(y_test, y_pred,"SVR")
 