from train import *
import pickle

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
    return model

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
    FF_model_trained = train_model(model, train_loader, criterion, optimizer, num_epochs)
    
    #save the model
    file = open("FF.save","wb")
    pickle.dump(FF_model_trained,file)
    file.close()
    
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
    FF_model_trained = train_model(model, train_loader, criterion, optimizer, num_epochs)
    
    # save the model
    file = open("FFplus.save","wb")
    pickle.dump(FF_model_trained,file)
    file.close()
    # Valutazione del modello
    return (evaluate_model(model, test_loader))

def FF_train(X_train,y_train,X_test,y_test):
    (train_loader, test_loader) =  prepare_model(X_train,y_train,X_test,y_test)
    metrics = FF_Model(PCA_X_train=X_train,train_loader=train_loader,test_loader=test_loader)
    return metrics

def FFplus_train(X_train,y_train,X_test,y_test):
    (train_loader, test_loader) =  prepare_model(X_train,y_train,X_test,y_test)
    metrics = FFPlus_model(PCA_X_train=X_train,train_loader=train_loader,test_loader=test_loader)
    return metrics
#### 