import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import argparse
import os
import json

def prepare_data(X, window_size, sc=None):
    '''
    window_size : number of timesteps to look back for predicting the next timestep
    '''
    nAssets=X.shape[1]
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
    
    return x.reshape(-1, nAssets * window_size), y, sc

def make_grid_plot(y,preds):
    fig, axs = plt.subplots(5, 6)
    for i in range(5):
        for j in range(6):
            idx=i*5+j
            axs[i,j].plot(preds[:, idx])
            axs[i,j].plot(sc.inverse_transform(y)[:, idx])
    return fig

#### Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-ename", "--exp_name", type=str, default="default_exp",
                    help="Expects a path to training folder")
parser.add_argument("-tsize", "--test_size", type=float, default=0.1,
                    help="Test Size")
parser.add_argument("-ws", "--window_size", type=int, default=5,
                    help="Window Size")
parser.add_argument("-nstocks", "--num_stocks", type=int, default=4,
                    help="Number of stocks to select")
args = parser.parse_args()


EXP_NAME=args.exp_name
# Make experiment directory #
if not os.path.exists(args.exp_name):
	os.makedirs(args.exp_name)

# Reading data
data=pd.read_csv('data2.csv')
data=np.array(data.iloc[:, :])

# Split into training, validation and testing parts
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(data, test_size=args.test_size, shuffle=False)
_xtrain=np.copy(X_train)
X_test=np.vstack([X_train[X_train.shape[0]-args.window_size:X_train.shape[0], :], X_test])

xtr,ytr,sc=prepare_data(X_train, args.window_size)
xte,yte,_=prepare_data(X_test, args.window_size, sc)
reg = LinearRegression().fit(xtr, ytr)
preds=sc.inverse_transform(reg.predict(xte))

# Pick top 4 best stocks #
num_stocks=X_train.shape[1]
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

with open(os.path.join(EXP_NAME, 'commandline_args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)


# Save predictions on test set along with other data #
arr=np.vstack([_xtrain, preds])
print(arr.shape)
np.save(os.path.join(EXP_NAME, 'preds_array.npy'), arr)

preds_tr=sc.inverse_transform(reg.predict(xtr))
train_fig=make_grid_plot(ytr,preds_tr)
test_fig=make_grid_plot(yte,preds)
train_fig.savefig(os.path.join(EXP_NAME, 'train_plot.png'))
test_fig.savefig(os.path.join(EXP_NAME, 'test_plot.png'))
 
