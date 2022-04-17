import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
import os
import argparse
import json

def prepare_data(X, window_size, sc=None):
    '''
    window_size : number of timesteps to look back for predicting the next timestep
    '''
    if sc is None:
        sc = MinMaxScaler()
        X = sc.fit_transform(X)
    else:
        X=sc.transform(X)
    
    x=[]
    y=[]
    dim=X.shape[1]
    for i in range(window_size, X.shape[0]):
        x.append(X[(i-window_size):i, :].reshape(window_size, dim))
        y.append(X[i, :].reshape(1, dim))
    
    x=np.array(x); y=np.squeeze(np.array(y));
    
    return Variable(torch.Tensor(x)), Variable(torch.Tensor(y)), sc

def make_grid_plot(x,y,lstm):
    fig, axs = plt.subplots(5, 6)
    preds=predict(x, sc, lstm)
    for i in range(5):
        for j in range(6):
            idx=i*5+j
            axs[i,j].plot(preds[:, idx])
            axs[i,j].plot(sc.inverse_transform(y)[:, idx])
    return fig


class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        h_out = h_out.view(-1, self.hidden_size)
        
        out = self.fc(h_out)
        
        return out


def predict(x, sc, model):
    model=model.eval()
    outputs = model(x)
    outputs=sc.inverse_transform(model(x).data.numpy())
    return outputs

#### Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-ename", "--exp_name", type=str, default="default_exp",
                    help="Expects a path to training folder")
parser.add_argument("-tsize", "--test_size", type=float, default=0.1,
                    help="Test Size")
parser.add_argument("-ws", "--window_size", type=int, default=5,
                    help="Batch Size")
parser.add_argument("-e", "--epochs", type=int, required=True,
                    help="Epochs")
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-2,
                    help="Learning Rate of non-embedding params")
parser.add_argument("-hsize", "--hidden", type=int, default=200,
                    help="Hidden Layer Size")
parser.add_argument("-nstocks", "--num_stocks", type=int, default=4,
                    help="Number of stocks to select")
args = parser.parse_args()



# Hyperparameters #
RANDOM_SEED = 42
train_size=1.0-2.0*(args.test_size)
val_size=0.5 # Decide between validation and testing
window_size=args.window_size
num_epochs = args.epochs
learning_rate = args.learning_rate
input_size = 32
hidden_size = args.hidden
num_layers = 1
num_classes = 32
num_stocks=args.num_stocks
EXP_NAME=args.exp_name

# Make experiment directory #
if not os.path.exists(args.exp_name):
	os.makedirs(args.exp_name)

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

# Reading data
data=pd.read_csv('data2.csv')
data=np.array(data.iloc[:, :])

# Split into training, validation and testing part
X_train, X_tem = train_test_split(data, test_size=1-train_size, shuffle=False)
X_val, X_test = train_test_split(X_tem, test_size=1-val_size, shuffle=False)
# Store a copy for future use #
_xtrain=np.copy(X_train)
_xval=np.copy(X_val)
_xtest=np.copy(X_test)

# Prepare data #
# Append last `window_size` timesteps to val/test
X_val=np.vstack([X_train[X_train.shape[0]-window_size:X_train.shape[0], :], X_val])
X_test=np.vstack([X_val[X_val.shape[0]-window_size:X_val.shape[0], :], X_test])
x_train,y_train,sc=prepare_data(X_train, window_size)
x_val,y_val,sc=prepare_data(X_val, window_size, sc)
x_test,y_test,sc=prepare_data(X_test, 5, sc)

# Train #
lstm = LSTM(num_classes, input_size, hidden_size, num_layers)
model_final=None
min_val_loss=1e20

criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

train_losses=[]
val_losses=[]
# Train the model
for epoch in range(num_epochs):
    outputs = lstm(x_train)
    optimizer.zero_grad()
    
    # obtain the loss function
    loss = criterion(outputs, y_train)
    train_losses.append(loss)
    lstm=lstm.eval()
    val_loss = criterion(lstm(x_val), y_val)
    lstm=lstm.train()
    val_losses.append(val_loss)
    
    loss.backward()
    
    optimizer.step()
    if epoch % 100 == 0:
        lstm=lstm.eval()
        val_loss = criterion(lstm(x_val), y_val)
        
        if val_loss <= min_val_loss:
            min_val_loss=val_loss
            model_final=lstm
            print("Model Updated")
        print("Epoch: %d, train loss: %1.5f, val loss: %1.5f" % (epoch, loss.item(), val_loss.item()))
        lstm=lstm.train()

train_fig=make_grid_plot(x_train,y_train,model_final)
val_fig=make_grid_plot(x_val,y_val,model_final)
test_fig=make_grid_plot(x_test,y_test,model_final)

train_fig.savefig(os.path.join(EXP_NAME, 'train_plot.png'))
val_fig.savefig(os.path.join(EXP_NAME, 'val_plot.png'))
test_fig.savefig(os.path.join(EXP_NAME, 'test_plot.png'))

# Pick top 4 best stocks #
num_stocks=X_train.shape[1]
preds=predict(x_test, sc, model_final)
print(preds.shape)
X=preds
R = np.copy(X)
for i in range(1, R.shape[0]):
    R[i] = (X[i] - X[i-1])/X[i-1]
    
R = R[1:, :]

expected_returns={}
for i in range(num_stocks):
    expected_returns[i]=np.mean(R[:, i])

top_idxs=list(dict(sorted(expected_returns.items(), key=lambda item: item[1], reverse=True)).keys())[:args.num_stocks]
np.save(os.path.join(EXP_NAME, 'top_idxs.npy'), np.array(top_idxs))

torch.save(model_final.state_dict(), os.path.join(EXP_NAME, 'model.pt'))

with open(os.path.join(EXP_NAME, 'commandline_args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)


# Save predictions on test set along with other data #
arr=np.vstack([_xtrain, _xval, preds])
print(arr.shape)
np.save(os.path.join(EXP_NAME, 'preds_array.npy'), arr)




fig, ax = plt.subplots(1)
ax.plot(np.linspace(1, len(train_losses), len(train_losses)), train_losses)
fig.savefig(os.path.join(EXP_NAME, 'train_loss.png'))
fig, ax = plt.subplots(1)
ax.plot(np.linspace(1, len(val_losses), len(val_losses)), val_losses)
fig.savefig(os.path.join(EXP_NAME, 'val_loss.png'))
