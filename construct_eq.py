import numpy as np
import pandas as pd
import cvxopt
import os
import json
import cvxopt as opt
from cvxopt import blas, solvers
from cvxopt import matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Hyperparameters #
base_folder='SVR_Data2'
target_return=0.2
out_folder='summary_eq2/svr'

def arr_to_ret(X):
    R = np.copy(X)
    for i in range(1, R.shape[0]):
        R[i] = (X[i] - X[i-1])/X[i-1]

    R = R[1:, :]
    return R

if not os.path.exists(out_folder):
    os.makedirs(out_folder)

returns_p=[]
risks_p=[]
srs_p=[]
returns_actual=[]
risks_actual=[]
srs_actual=[]

for folder_name in os.listdir(base_folder):
    folder=os.path.join(base_folder, folder_name)

    with open(os.path.join(folder, 'commandline_args.txt')) as file:
        d=json.loads(file.read())
    idxs=np.load(os.path.join(folder, 'top_idxs.npy'))
    num_stocks=d['num_stocks']
    idxs=idxs[:num_stocks]

    preds=np.load(os.path.join(folder, 'preds_array.npy'))
    data=np.array(pd.read_csv('data2.csv').iloc[:, :])
    assert preds.shape==data.shape
    preds=preds[:, idxs]
    data=data[:, idxs]

    # Test data average return compuation #
    data, test_data = train_test_split(data, test_size=d['test_size'], shuffle=False)
    test_mu=np.mean(arr_to_ret(test_data), axis=0).reshape(1, num_stocks)

    R=arr_to_ret(preds)
    R_m = R - np.mean(R, axis=0)
    omega = np.dot(R_m.T, R_m)/(np.shape(R_m)[0])
    w_opt=np.ones((num_stocks, 1))/num_stocks

    risk=np.dot(w_opt.T, np.dot(omega, w_opt))[0][0]
    ret=np.dot(test_mu, w_opt)[0][0]
    sr=ret/risk
    risks_p.append(risk)
    returns_p.append(ret)
    srs_p.append(sr)

    
    R=arr_to_ret(data)
    R_m = R - np.mean(R, axis=0)
    omega = np.dot(R_m.T, R_m)/(np.shape(R_m)[0])
    w_opt=np.ones((num_stocks, 1))/num_stocks

    risk=np.dot(w_opt.T, np.dot(omega, w_opt))[0][0]
    ret=np.dot(test_mu, w_opt)[0][0]
    sr=ret/risk
    risks_actual.append(risk)
    returns_actual.append(ret)
    srs_actual.append(sr)

plt.figure()
plt.plot(returns_p)
plt.plot(returns_actual)
plt.legend(["predicted", "actual"], loc ="lower right")
plt.title('Returns')
plt.savefig(os.path.join(out_folder, 'returns.png'))

plt.figure()
plt.plot(risks_p)
plt.plot(risks_actual)
plt.legend(["predicted", "lstm_actual"], loc ="lower right")
plt.title('Risks')
plt.savefig(os.path.join(out_folder, 'risks.png'))

plt.figure()
plt.plot(srs_p)
plt.plot(srs_actual)
plt.legend(["predicted", "actual"], loc ="lower right")
plt.title('SR')
plt.savefig(os.path.join(out_folder, 'sr.png'))
