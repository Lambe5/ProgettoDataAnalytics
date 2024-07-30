from train import *

def linearReg(X_train, y_train, X_test, y_test):
    #Linear-Regressor
    reg = LinearRegression().fit(X_train, y_train)
    
    #TODO: Salvare il file del modello con pickle
    file = open("Linear_regression.save","wb")
    pickle.dump(reg,file)
    file.close()

    predictions = reg.predict(X_test)

    metrics = calc_metrics(y_test,predictions,"linear_Regression")
    return metrics
