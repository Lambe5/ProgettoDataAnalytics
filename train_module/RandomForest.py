from test import *
def randomForest(X_train, y_train, X_test, y_test, n_alberi):
    rf_regressor = RandomForestRegressor(n_estimators=400,#TODO: da verificare iper parametri
                                                  max_depth= 140,
                                                  min_samples_split= 15,
                                                   random_state=42, n_jobs = -1)
    rf_regressor.fit(X_train, y_train)
    #TODO: Salvare il file del modello con pickle
    file = open("RandomForest.save","wb")
    pickle.dump(rf_regressor,file)
    file.close()
    rf_predictions = rf_regressor.predict(X_test)
    return calc_metrics(y_test,rf_predictions,"RandomForest")
