# Code for generating experiments, analysis, and plots used in the Climate Lobes publication
# For the vary-lobe experiment: Use 100% of training data from one lobe of L63, and a variable percentage from the other lobe
# For the ESN variants (ESN-LSR and ESN-HSR)

from scipy.stats import wasserstein_distance
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
import pickle

######### Control which experimental procedure is run here
doWasser= False # if True: Runs Wasserstein Distance Experiment. If False: Runs Trajectory Plotting Experiment
useHSR = True# if True: Runs previously selected experiment with HSR-ESN. If False: Runs previously selected experiment with LSR-ESN
##########



if useHSR: # Settings for the HSR-ESN
    rho_use = 1.2
    degree_use = 6
    sigma_use = .1
    regularize = False
else: # assume LSR otherwise: Settings for the LSR-ESN
    rho_use = .1
    degree_use = 3
    sigma_use = .5
    regularize = True

np.random.seed(1800)

if doWasser:
    Ntrain = 50000
    Ntest = 100000 # long term climate (wasserstein metric)
    Ntrial = 5 # 5 for wasserstein
else:
    Ntrain = 50000
    Ntest = 2000 # Shorter trajectory predictions for plotting (non-Wasserstein) variant
    Ntrial = 25 # 25 for plotting

doActivations = False
if doActivations: # Smaller trials just to test out activation function statistics
    Ntest = 2000
    Ntrial = 2

actfn = lambda x: np.tanh(x)
actdrstore = np.array([])

def nonlin(x): # odd squaring function
    x2 = np.copy(x)
    if len(np.shape(x2))==2:
        for i in range(np.shape(x)[1] // 2):
            x2[:, 2 * i] = (x2[:, 2 * i] ** 2).copy()
        return x2
    else: # assuming len = 1
        for i in range(len(x2) // 2):
            x2[2 * i] = (x2[2 * i] ** 2).copy()
        return x2

# Standard settings for a Chaotic L63 attractor
rho_l = 28.0
sigma_l = 10.0
beta_l = 8.0 / 3.0
dim = 3

def f(state, t):
    x, y, z = state  # Unpack the state vector
    return sigma_l * (y - x), x * (rho_l - z) - y, x * y - beta_l * z  # Derivatives

# Generate Training and Testing Data
state0 = [1.0, 1.0, 1.0]
t = np.linspace(0.0, (Ntrain+Ntest*Ntrial)/200 + .005, Ntrain+(Ntest*Ntrial+1))
tinv = np.linspace(0,-50/200, 51)
states = odeint(f, state0, t)

useFE = False # use manual forward euler if True
if (useFE):
    states[0,:] = state0 + .005*np.array(f(state0,0))
    for i in range(Ntrain + Ntest*Ntrial):
        states[i+1,:] = states[i,:] + .005*np.array(f(states[i,:],0))

regularized = np.std(states,0)
statemean = np.mean(states,0)
if regularize:
    states /= regularized # scaled

qerror = np.zeros((Ntrial, Ntest))
Wasser = np.zeros((Ntrial,4))
rankstore = np.zeros(Ntrial)

for q in range(Ntrial):
    print(q)
    statesuse = states[q * Ntest:Ntrain + q * Ntest + Ntest, :]
    trainper = True
    leftrate = 1.0
    rightrate = 1.0
    if trainper:
        leftdist  = np.sqrt((statesuse[:-Ntest, 0] - np.sqrt(72))**2 + (statesuse[:-Ntest, 1] - np.sqrt(72))**2 + (statesuse[:-Ntest, 2] - 27)**2)
        rightdist = np.sqrt((statesuse[:-Ntest, 0] + np.sqrt(72))**2 + (statesuse[:-Ntest, 1] + np.sqrt(72))**2 + (statesuse[:-Ntest, 2] - 27)**2)
        leftuse = np.where(leftrate* leftdist < rightdist)[0]
        rightuse = np.where(leftrate * leftdist > rightdist)[0]
        # NEW - get leftuse, rightuse into 'segments of minimum length (say 50?)
        # Can do this by breaking up leftuse/rightuse?
        leftstart = leftuse[0]
        leftseqs = []
        rightstart = rightuse[0]
        rightseqs = []
        for i in range(1,len(leftuse)):
            if leftuse[i] != leftuse[i-1]+1: # Mismatch - jumping from left to right
                leftseqs.append([leftstart,leftuse[i-1]])
                leftstart = leftuse[i]
        for i in range(1, len(rightuse)):
            if rightuse[i] != rightuse[i - 1] + 1:  # Mismatch - jumping from left to right
                rightseqs.append([rightstart, rightuse[i - 1]])
                rightstart = rightuse[i]
        # Process - remove entires that start before 50, or have minimum length less than 50
        pc = 0
        for i in range(len(leftseqs)):
            if leftseqs[i-pc][0] < 50:
                leftseqs.pop(i-pc)
                pc+=1
            if leftseqs[i-pc][1]-leftseqs[i-pc][0]<50:
                leftseqs.pop(i-pc)
                pc+=1
        pc = 0
        for i in range(len(rightseqs)):
            if rightseqs[i-pc][0] < 50:
                rightseqs.pop(i-pc)
                pc+=1
            if rightseqs[i-pc][1]-rightseqs[i-pc][0]<50:
                rightseqs.pop(i-pc)
                pc+=1
        # Now we have a list of start/stop sequences to train on in each lobe
        # Decide to train on each one using left rate / right rate

        # Reservoir setup - can be tuned to either HSR or LSR
        N = 100  # 100 default
        p = np.min([degree_use / N, 1.0])
        rho = rho_use
        A = np.random.rand(N, N)
        Amask = np.random.rand(N, N)
        A[Amask > p] = 0
        [eigvals, eigvecs] = np.linalg.eig(A)
        A /= (np.real(np.max(eigvals)))
        A *= rho
        sigma = sigma_use
        Win = np.random.rand(dim, N) * 2 - 1  # dense
        Win = Win * 0
        Win[0, :N // 3] = (np.random.rand(N // 3) * 2 - 1) * sigma
        Win[1, N // 3:2 * N // 3] = (np.random.rand(2 * N // 3 - N // 3) * 2 - 1) * sigma
        Win[2, 2 * N // 3:] = (np.random.rand(N - 2 * N // 3) * 2 - 1) * sigma

        r = np.zeros(N)
        rout = np.zeros((N, 1))
        traininds = []  # store indices used so owe can train on them

        # Begin training - left
        for i in range(len(leftseqs)):
            if np.random.rand() < leftrate:
                # Warmup first
                for j in range(max(leftseqs[i][0] - 50, 0), leftseqs[i][0]):
                    r = actfn(A @ r + (statesuse[j, :]) @ Win)
                # Then real training
                for j in range(leftseqs[i][0], leftseqs[i][1]):
                    activation = A @ r + (statesuse[j, :]) @ Win
                    actdr = 1 / (np.cosh(activation) ** 2)
                    actdrstore = np.append(actdrstore, actdr)
                    r = actfn(activation)
                    rout = np.c_[rout, r]
                    traininds.append(j)
        # Same thing with right
        for i in range(len(rightseqs)):
            if np.random.rand() < rightrate:
                # Warmup first
                for j in range(max(rightseqs[i][0] - 50, 0), rightseqs[i][0]):
                    r = actfn(A @ r + (statesuse[j, :]) @ Win)
                # Then real training
                for j in range(rightseqs[i][0], rightseqs[i][1]):
                    activation = A @ r + (statesuse[j, :]) @ Win
                    actdr = 1 / (np.cosh(activation) ** 2)
                    actdrstore = np.append(actdrstore, actdr)
                    r = actfn(activation)
                    rout = np.c_[rout, r]
                    traininds.append(j)

        # Train matrix
        rout = rout[:, 1:]
        rout = rout.transpose()
        trout = nonlin(rout)
        # Test rank of trout
        rankstore[q] = np.linalg.matrix_rank(trout)

        # Train offline  - min error norm + l2 error of trout*Wout - states, where Wout = Nx3
        Id_n = np.identity(N)
        beta = .0001
        U = np.dot(trout.transpose(), trout) + Id_n * beta
        Uinv = np.linalg.inv(U)
        Wout = np.dot(Uinv, np.dot(trout.transpose(), statesuse[np.array(traininds) + 1, :]))

        # Predictions - unroll reservoir for Ntest
        r = np.zeros(N)
        rpred2 = np.zeros((Ntest, N))
        for i in range(Ntrain - 50, Ntrain):
            r = actfn(A @ r + (statesuse[i, :]) @ Win)
        r2 = np.copy(r)
        for i in range(Ntrain, Ntrain + Ntest):
            r3 = nonlin(r2)
            r2 = actfn(A @ r2 + r3 @ Wout @ Win)
            rpred2[i - Ntrain, :] = r2
        trpred2 = nonlin(rpred2)
        ResPred2 = trpred2 @ Wout
        errors2 = np.sqrt(np.sum((ResPred2 - statesuse[-Ntest:,:]) ** 2, 1))
        if regularize:
            errors2 = np.sqrt(np.sum((ResPred2*regularized - statesuse[-Ntest:,:]*regularized)**2,1))
        qerror[q, :] = errors2

        if (regularize):
            ResPred2*=regularized
            states*=regularized
        Wasser[q, 0] = wasserstein_distance(ResPred2[:, 0], states[1000:, 0])
        Wasser[q, 1] = wasserstein_distance(ResPred2[:, 1], states[1000:, 1])
        Wasser[q, 2] = wasserstein_distance(ResPred2[:, 2], states[1000:, 2])
        # Do random projections to get Wasser[q,3]
        numWasser = 100
        projWass = np.zeros(numWasser)
        loadWasser = True
        if loadWasser:
            import pickle
            file_proj = open('.\ESN\Projs.pckl', 'rb')
            test2 = pickle.load(file_proj)
            projs = test2
        for i in range(numWasser):
            if doWasser:
                proj = projs[:,i]
                proj /= np.linalg.norm(proj)
                projWass[i] = wasserstein_distance((ResPred2 @ proj).flatten(), (states[1000:, :] @ proj).flatten())
            Wasser[q, 3] = np.mean(projWass)
        if (regularize):
            states/=regularized
            ResPred2/=regularized

# Save data of interest - Wasser, qerror, actdrstore

savedat = False
path = '' # Path and filename to store
if savedat:
    with open(path, 'wb') as f:
            pickle.dump([actdrstore, qerror, Wasser], f)
