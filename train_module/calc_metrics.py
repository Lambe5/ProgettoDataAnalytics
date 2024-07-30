def calc_metrics(y_true,y_pred,model="n/s"):
    return {
            "mae_": mean_absolute_error(y_true,y_pred),
            "mse_": mean_squared_error(y_true, y_pred),
            "mape_": mean_absolute_percentage_error(y_true, y_pred),
            "r2score_": r2_score(y_true, y_pred),
            "model": model
        }