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
out_folder='summary2/svr'

def arr_to_ret(X):
	R = np.copy(X)
	for i in range(1, R.shape[0]):
		R[i] = (X[i] - X[i-1])/X[i-1]

	R = R[1:, :]
	return R

def markowitz_no_constraint(X):
    R = np.copy(X)
    for i in range(1, R.shape[0]):
        R[i] = (X[i] - X[i-1])/X[i-1]

    R = R[1:, :]

    # Solve optimiation problem #
    R_m = R - np.mean(R, axis=0)
    omega = np.dot(R_m.T, R_m)/(np.shape(R_m)[0])

    # No target return constraint #
    nAssets = omega.shape[0]
    e = np.ones((nAssets, 1))
    u = np.dot(np.linalg.inv(omega), e)
    w_opt = u/(np.dot(e.T, u))
    assert np.allclose(np.sum(w_opt), 1.0), "w_opt not summing to 1.0"
    mu = np.mean(R, axis=0).reshape(-1, 1)
    
    portfolio_return = np.sum(np.dot(mu.T, w_opt))
    portfolio_risk = np.dot(w_opt.T, np.dot(omega, w_opt))[0][0]
    sharpe_ratio=portfolio_return/portfolio_risk
    
    return portfolio_return, portfolio_risk, sharpe_ratio, w_opt

def markowitz_target_return(X, target_return):
    R = np.copy(X)
    for i in range(1, R.shape[0]):
        R[i] = (X[i] - X[i-1])/X[i-1]

    R = R[1:, :]

    # Solve optimiation problem #
    R_m = R - np.mean(R, axis=0)
    omega = np.dot(R_m.T, R_m)/(np.shape(R_m)[0])
    
    """
    1/2 x'Px + q'x
    subject to :
    Gx <= h
    Ax = b

    sol = solvers.qp(P,q,G,h,A,b)
    """
    nAssets = omega.shape[0]
    e = np.ones(nAssets).reshape(1, -1)
    mu = np.mean(R, axis=0).reshape(1, -1)

    P = matrix(2.0 * omega, tc='d') # tc='d' for double matrices
    q = matrix(np.zeros(nAssets))
    G = matrix(np.zeros((nAssets, nAssets)), tc='d')
    h = matrix(np.zeros(nAssets), tc='d')
    A = matrix(np.vstack([e, mu]), tc='d')
    b = matrix([1.0, target_return])

    sol = solvers.qp(P,q,G,h,A,b)

    w_opt = np.array(sol['x'])

    assert np.allclose(np.sum(w_opt), 1.0), "w_opt not summing to 1.0"
    assert np.allclose(np.dot(mu, w_opt), target_return), "mu'w not 0.2"
    
    portfolio_return = np.sum(np.dot(mu, w_opt))
    portfolio_risk = np.dot(w_opt.T, np.dot(omega, w_opt))[0][0]
    sharpe_ratio=portfolio_return/portfolio_risk
    
    return portfolio_return, portfolio_risk, sharpe_ratio, w_opt


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

	_, risk, _, w_opt = markowitz_no_constraint(preds)
	ret=np.dot(test_mu, w_opt)[0][0]
	sr=ret/risk
	risks_p.append(risk)
	returns_p.append(ret)
	srs_p.append(sr)

	
	print(data.shape)
	_, risk, _, w_opt = markowitz_no_constraint(data)
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
