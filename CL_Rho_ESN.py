# Code for generating experiments, analysis, and plots used in the Climate Lobes publication
# For the vary-rho experiment: Train on some rho values, then test on values inside of or outside of testing range
# For the ESN variants (ESN-LSR and ESN-HSR)

from scipy.stats import wasserstein_distance
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations_with_replacement
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
import pickle

Ntrainsm = 1000
Ntrainsm = 2500
Ntrain = 50000
Ntrain = Ntrainsm
Ntest = 100000 + (50000-Ntrainsm) # longer, for wasser. Just only use first 2k for error/output plots
Ntrial = 1
Ntrialuse = 1

rho_interp = 23.5
rho_extrap = 28

rho_test_interp = [16,20,25]
rho_test_extrap = [8,10,26,30,34]

regularize = False
useHSR = True
useHSR = False # LSR
if useHSR:
    rho_use = 1.2
    degree_use = 6
    sigma_use = .1
else: # assume LSR otherwise
    rho_use = .1
    degree_use = 3
    sigma_use = .5
    regularize = True

actfn = lambda x: np.tanh(x)

def nonlin(x): # odd squaring
    x2 = np.copy(x)
    if len(np.shape(x2))==2:
        for i in range(np.shape(x)[1] // 2):
            x2[:, 2 * i] = (x2[:, 2 * i] ** 2).copy()
        return x2
    else: # assuming len = 1
        for i in range(len(x2) // 2):
            x2[2 * i] = (x2[2 * i] ** 2).copy()
        return x2

Wasser = np.zeros((Ntrialuse, 2))
for q in range(Ntrialuse):
    print(q)
    rho_l_use = np.linspace(12,24,20)
    for i in range(len(rho_l_use)):
        rho_l = rho_l_use[i]
        sigma_l = 10.0
        beta_l = 8.0 / 3.0
        dim = 3
        def f(state,t):
            x, y, z = state  # Unpack the state vector
            return sigma_l * (y - x), x * (rho_l - z) - y, x * y - beta_l * z  # Derivatives
        tsm = np.linspace(0.0, (Ntrainsm*Ntrial)/200 + .005, Ntrainsm+1)
        tsminv = np.linspace(0, -(50)/200, 51)
        stateinit = [1.0, 1.0, 1.0]
        statetrain = odeint(f,stateinit,tsm) # Ntrainsm + 1 by 3
        if regularize:
            regularized = np.std(statetrain, 0)
            statetrain/=regularized
        if i == 0:
            traindat = statetrain
        else:
            traindat = np.append(traindat, statetrain, axis=0)

    state0 = [1.0, 1.0, 1.0]
    t = np.linspace(0.0, (Ntrain+Ntest*Ntrial)/200 + .005, Ntrain+(Ntest*Ntrial)+1)

    rho_l = rho_interp
    def f(state,t):
        x, y, z = state  # Unpack the state vector
        return sigma_l * (y - x), x * (rho_l - z) - y, x * y - beta_l * z  # Derivatives
    states_interp = odeint(f, state0, t)

    rho_l = rho_extrap
    def f(state,t):
        x, y, z = state  # Unpack the state vector
        return sigma_l * (y - x), x * (rho_l - z) - y, x * y - beta_l * z  # Derivatives
    states_extrap = odeint(f, state0, t)

    if regularize:
        regularized_interp = np.std(states_interp, 0)
        states_interp /= regularized_interp  # scaled
        regularized_extrap = np.std(states_extrap, 0)
        states_extrap /= regularized_extrap  # scaled

    # Setup Reservoir
    N = 100  # 100 default
    p = np.min([degree_use / N, 1.0])
    rho = rho_use
    A = np.random.rand(N, N)
    Amask = np.random.rand(N, N)
    A[Amask > p] = 0
    [eigvals, eigvecs] = np.linalg.eig(A)
    A /= (np.real(np.max(eigvals)))
    A *= rho
    A *= 1
    sigma = sigma_use
    Win = np.random.rand(dim, N) * 2 - 1  # dense
    Win = Win * 0
    Win[0, :N // 3] = (np.random.rand(N // 3) * 2 - 1) * sigma
    Win[1, N // 3:2 * N // 3] = (np.random.rand(2 * N // 3 - N // 3) * 2 - 1) * sigma
    Win[2, 2 * N // 3:] = (np.random.rand(N - 2 * N // 3) * 2 - 1) * sigma
    r = np.zeros(N)
    rout = np.zeros((N, 1))
    # Begin training: For each len 501 fragment, need to get a previous warmup states
    # Then get the actual reservoir activations
    for i in range(len(rho_l_use)):
        trainstateinit = traindat[i*(Ntrainsm+1),:]
        statesinv = odeint(f, trainstateinit, tsminv)
        r = np.zeros(N)
        for j in range(50):
            r = actfn(A @ r + (statesinv[50-j, :]) @ Win)
        for j in range(Ntrainsm):
            r = actfn(A @ r + (traindat[i*(Ntrainsm+1)+j, :]) @ Win)
            rout = np.c_[rout, r]
    # Generate Training Matrix, Train
    rout = rout[:, 1:]
    rout = rout.transpose()
    trout = nonlin(rout)

    # Augment with rho information
    for i in range(len(rho_l_use)):
        rhodat = np.ones(Ntrainsm) * rho_l_use[i]
        if i == 0:
            rhoaug = rhodat
        else:
            rhoaug = np.append(rhoaug, rhodat)
    augtrout = np.vstack([np.transpose(trout), rhoaug])

    for i in range(len(rho_l_use)):
        if i == 0:
            traintargets = traindat[1:(Ntrainsm+1), :]
        else:
            traintargets = np.vstack([traintargets, traindat[(Ntrainsm+1) * i + 1:(Ntrainsm+1) * i + (Ntrainsm+1), :]])

    # Train offline  - min error norm + l2 error of trout*Wout - states, where Wout = Nx3
    Id_n = np.identity(N+1) # +1 from augmentation
    beta = .0001
    U = np.dot(augtrout, augtrout.transpose()) + Id_n * beta
    Uinv = np.linalg.inv(U)
    Wout = np.dot(Uinv, np.dot(augtrout, traintargets))


    # Test on both outputs: state24 and state28
    # Both will require a warmup, but fortunately data is available already
    r = np.zeros(N)
    rout1 = np.zeros((Ntest, N+1))
    rtest1 = np.zeros((50,N+1))
    for i in range(50):
        r = actfn(A @ r + (states_interp[-Ntest-50 -1 + i,:]) @ Win)
        rtest1[i,:N] = r
        rtest1[i,-1] = rho_interp
        # Diagnostic * nonlin(rtest1) @ Wout should be equal to states24[-Ntest-50:-Ntest,:]
        # it is - starts out moderately innacurate (as expected), but by i = 50, error is order 1e-2
        # Diagnostic : nonlin(rtest1) should be similar to trout[-50:, :]??
        # Highly accurate - delta is .00xx, so not perfect but excellent. More or less starting from same 'state'...
    for i in range(Ntest):
        r2 = nonlin(r)
        r = actfn(A @ r + np.append(r2,rho_interp) @ Wout @ Win) # 23.85, rho_l_use[-2]
        rout1[i, :N] = r
    trout1 = nonlin(rout1)
    trout1[:,-1] = rho_interp
    ResPred_interp = trout1 @ Wout
    if regularize:
        ResPred_interp*=regularized_interp
        states_interp*=regularized_interp
    errors2_res_interp = np.sqrt(np.sum((ResPred_interp - states_interp[-Ntest:, :]) ** 2, 1))

    r = np.zeros(N)
    rout2 = np.zeros((Ntest, N+1))
    for i in range(50):
        r = actfn(A @ r + (states_extrap[-Ntest - 50 -1  + i, :]) @ Win)
    for i in range(Ntest):
        r2 = nonlin(r)
        r = actfn(A @ r + np.append(r2,rho_extrap) @ Wout @ Win)
        rout2[i, :N] = r
        #rout2[-1, i] = 28
    trout2 = nonlin(rout2)
    trout2[:,-1] = rho_extrap
    ResPred_extrap = trout2 @ Wout
    if regularize:
        ResPred_extrap*=regularized_extrap
        states_extrap*=regularized_extrap
    errors2_res_extrap = np.sqrt(np.sum((ResPred_extrap - states_extrap[-Ntest:, :]) ** 2, 1))

    # Check Wasser, do plots, etc
    if q==0:
        plt.plot(errors2_res_interp[:2000])
        plt.title('Testing Trajectory Error, ESN, Interpolation')
        plt.show()

        plt.plot(errors2_res_extrap[:2000])
        plt.title('Testing Trajectory Error, ESN, Extrapolation')
        plt.show()

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(ResPred_interp[:2000, 0], ResPred_interp[:2000, 1], ResPred_interp[:2000, 2], 'r', alpha = .5)
        ax.plot(states_interp[-Ntest:-Ntest + 2000, 0], states_interp[-Ntest:-Ntest + 2000, 1], states_interp[-Ntest:-Ntest + 2000, 2], 'b', alpha = .5)
        plt.title('Testing Trajectories, ESN, Interpolation', fontsize=16)
        plt.legend(['Predicted', 'True'], fontsize=16, loc=3)
        plt.xlabel('X', fontsize=16)
        plt.ylabel('Y', fontsize=16)
        ax.set_zlabel('Z', fontsize=16)
        plt.draw()
        plt.show()

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(ResPred_extrap[:2000, 0], ResPred_extrap[:2000, 1], ResPred_extrap[:2000, 2], 'r', alpha = .5)
        ax.plot(states_extrap[-Ntest:-Ntest + 2000, 0], states_extrap[-Ntest:-Ntest + 2000, 1], states_extrap[-Ntest:-Ntest + 2000, 2],'b', alpha = .5)
        plt.title('Testing Trajectories, ESN, Extrapolation', fontsize=16)
        plt.legend(['Predicted', 'True'], fontsize=16, loc=3)
        plt.xlabel('X', fontsize=16)
        plt.ylabel('Y', fontsize=16)
        ax.set_zlabel('Z', fontsize=16)
        plt.draw()
        plt.show()


    numWasser = 100
    projWass = np.zeros(numWasser)
    projWass2 = np.zeros(numWasser)
    loadWasser = True
    if loadWasser:
        import pickle
        file_proj = open('.\ESN\Projs.pckl', 'rb')
        test2 = pickle.load(file_proj)
        projs = test2
    for i in range(numWasser):
        proj = projs[:, i]
        proj /= np.linalg.norm(proj)
        projWass[i] = wasserstein_distance((ResPred_interp @ proj).flatten(), (states_interp[-Ntest:, :] @ proj).flatten())
        projWass2[i] = wasserstein_distance((ResPred_extrap @ proj).flatten(), (states_extrap[-Ntest:, :] @ proj).flatten())
    Wasser[q, 0] = np.mean(projWass)
    Wasser[q, 1] = np.mean(projWass2)

# Save data of interest - Wasser

savedat = False
path = '' # Path and filename to store
if savedat:
    with open(path, 'wb') as f:
            pickle.dump([Wasser], f)