import pandas as pd
from sklearn.model_selection import train_test_split

FILENAME = "train.csv"
df = pd.read_csv(FILENAME)
df.shape

df.isna().sum()

df = df.drop_duplicates()
df.shape

# Split in Train, Val e Test

def train_test(dataset, save):
    X = dataset.iloc[:, 1:]
    y = dataset[["Year"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if (save):
        X_train.to_csv("./Datasets/Train_Test/X_train.csv", index=False)
        y_train.to_csv("./Datasets/Train_Test/y_train.csv", index=False)
        X_test.to_csv("./Datasets/Train_Test/X_test.csv", index=False)
        y_test.to_csv("./Datasets/Train_Test/y_test.csv", index=False)

    return (X_train, X_test, y_train, y_test)

def train_validation_test(dataset, save):

    (X_train, X_test, y_train, y_test) = train_test(dataset, save)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    X_train.to_csv("./Datasets/Train_Val_Test/X_train.csv", index=False)
    y_train.to_csv("./Datasets/Train_Val_Test/y_train.csv", index=False)
    X_test.to_csv("./Datasets/Train_Val_Test/X_test.csv", index=False)
    y_test.to_csv("./Datasets/Train_Val_Test/y_test.csv", index=False)
    X_val.to_csv("./Datasets/Train_Val_Test/X_val.csv", index=False)
    y_val.to_csv("./Datasets/Train_Val_Test/y_val.csv", index=False)

    return(X_train, X_val, y_train, y_val, X_test, y_test)

def load_df(path1, path2):
    X_train = pd.read_csv(path1)
    y_train = pd.read_csv(path2)
    return (X_train, y_train)

# Salva i dataset
#train_test(df, True)
#train_validation_test(df, False)