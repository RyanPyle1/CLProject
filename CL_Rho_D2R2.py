# Code for generating experiments, analysis, and plots used in the Climate Lobes publication
# For the vary-rho experiment: Train on some rho values, then test on values inside of or outside of testing range
# For the D2R2 algorithm


from scipy.stats import wasserstein_distance
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations_with_replacement
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
import pickle

Ntrainsm = 2500
Ntrain = 50000
Ntrain = Ntrainsm
Ntest = 100000 + (50000-Ntrainsm)# longer, for wasser. Just only use first 2k for error/output plots
Ntrial = 1
Ntrialuse = 1

rho_interp = 23.5
rho_extrap = 28

rho_test_interp = [16,20,25]
rho_test_extrap = [8,10,26,30,34]

userhoaug = True

Wasser = np.zeros((Ntrialuse, 2))  # only 1 trial - D2R2 will be deterministic based on data
for q in range(Ntrialuse):
    print(q)

    rho_l_use = np.linspace(12, 24, 20)
    for i in range(len(rho_l_use)):
        rho_l = rho_l_use[i]
        sigma_l = 10.0
        beta_l = 8.0 / 3.0
        dim = 3
        def f(state,t):
            x, y, z = state  # Unpack the state vector
            return sigma_l * (y - x), x * (rho_l - z) - y, x * y - beta_l * z  # Derivatives
        tsm = np.linspace(0.0, (Ntrainsm*Ntrial)/200 + .005, Ntrainsm+1)
        stateinit = np.random.rand(3)*10 - 20 # randomly in a large range
        statetrain = odeint(f,stateinit,tsm) # Ntrainsm + 1 by 3
        if i == 0:
            traindat = statetrain
        else:
            traindat = np.append(traindat, statetrain, axis=0)

    state0 = [1.0, 1.0, 1.0] + (np.random.rand(3)*2 - 1)*.1 # randomly +/- up to .1
    t = np.linspace(0.0, (Ntrain+Ntest*Ntrial)/200 + .005, Ntrain+(Ntest*Ntrial+1))

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

    # Train on the mini sets
    order = 4
    def polyfeat(states, order): # Given state vector, turn it into feature vector to fit linear regression with
        # We are using their convention that features[0] = 0, features[1] = polypred(states[0]) etc
        N = np.shape(states)[0]
        size = 1 + N  # 0 order will have 1 term, 1 order will have N terms
        for i in range(2, order + 1):
            comb = combinations_with_replacement(np.arange(N), i)
            combos = list(comb)
            size += len(combos)
        polyfeatures = np.zeros((size,np.shape(states)[1]))

        pp = PolynomialFeatures(degree=order)

        for i in range(np.shape(states)[1]):
            polyfeatures[:, i] = pp.fit_transform(states[:, i].reshape(1,-1))
        return polyfeatures, order

    # Augment with rho information
    for i in range(len(rho_l_use)):
        rhodat = np.ones(Ntrainsm+1)*rho_l_use[i]
        if i==0:
            rhoaug = rhodat
        else:
            rhoaug = np.append(rhoaug,rhodat)
    if not userhoaug:
        rhoaug*=0
    augstates = np.vstack([np.transpose(traindat), rhoaug])


    polyfeatures, order = polyfeat(augstates, order)
    pp = PolynomialFeatures(degree=order)

    for i in range(len(rho_l_use)):
        if i==0:
            polyfeattrain = augstates[:,:Ntrainsm]
            traintargets = traindat[1:Ntrainsm+1,:]
        else:
            polyfeattrain = np.hstack([polyfeattrain,augstates[:,(Ntrainsm+1)*i:(Ntrainsm+1)*i + Ntrainsm]])
            traintargets = np.vstack([traintargets, traindat[(1+Ntrainsm)*i+1:(1+Ntrainsm)*i+(Ntrainsm+1),:]])
    polyfeattrain, order = polyfeat(polyfeattrain, order)

    Id_n = np.identity(np.shape(polyfeattrain)[0]) # 70 x 70
    beta = .001 # original .001
    U = np.dot(polyfeattrain, polyfeattrain.transpose()) + Id_n * beta # 70 x 70, polyfeattrain is 70x50k
    Uinv = np.linalg.inv(U)
    Wout_poly = np.dot(Uinv, np.dot(polyfeattrain, traintargets)) #traintargets is 50k x 3
    sigma2 = np.var(polyfeattrain.transpose() @ Wout_poly)


    ResTrain_poly = polyfeattrain.transpose() @ Wout_poly
    errors_og_poly = np.sqrt(np.sum((ResTrain_poly - traintargets) ** 2, 1))
    # Test on 2 trajectories - one it has seen, one it has not
    # First test one something in its testing set

    laststate = states_interp[Ntrain+1, :]
    polyfeaturestest_interp = np.zeros((Ntest, 70))
    for i in range(Ntest):
        polyfeaturesuse = pp.fit_transform(np.append(laststate,(rho_interp)*(userhoaug)).reshape(1, -1)).squeeze()
        nextstate = polyfeaturesuse @ Wout_poly
        polyfeaturestest_interp[i, :] = polyfeaturesuse
        laststate = nextstate
    ResPred2_poly_interp = polyfeaturestest_interp @ Wout_poly
    errors2_poly_interp = np.sqrt(np.sum((ResPred2_poly_interp - states_interp[-Ntest:, :]) ** 2, 1))

    # better - still not perfect
    if q==0:
        plt.plot(errors2_poly_interp[:2000])
        plt.title('Testing Trajectory Error, D2R2, Interpolation')
        plt.show()

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(ResPred2_poly_interp[:2000, 0], ResPred2_poly_interp[:2000, 1], ResPred2_poly_interp[:2000, 2], 'r', alpha = .5)
        ax.plot(states_interp[-Ntest:-Ntest + 2000, 0], states_interp[-Ntest:-Ntest + 2000, 1], states_interp[-Ntest:-Ntest + 2000, 2], 'b', alpha = .5)
        plt.title('Testing Trajectories, D2R2, Interpolation', fontsize = 16)
        plt.legend(['Predicted', 'True'], fontsize = 16, loc = 3)
        plt.xlabel('X', fontsize = 16)
        plt.ylabel('Y', fontsize = 16)
        ax.set_zlabel('Z', fontsize = 16)
        #plt.zlabel('Z')
        plt.draw()
        plt.show()

    # Try again, on something it has never seen

    laststate = states_extrap[Ntrain+1, :]
    polyfeaturestest_extrap = np.zeros((Ntest, 70))
    for i in range(Ntest):
        polyfeaturesuse = pp.fit_transform(np.append(laststate,(rho_extrap)*(userhoaug)).reshape(1, -1)).squeeze()
        nextstate = polyfeaturesuse @ Wout_poly
        polyfeaturestest_extrap[i, :] = polyfeaturesuse
        laststate = nextstate
    ResPred2_poly_extrap = polyfeaturestest_extrap @ Wout_poly
    errors2_poly_extrap = np.sqrt(np.sum((ResPred2_poly_extrap - states_extrap[-Ntest:, :]) ** 2, 1))

    # Looks pretty decent. Lets examine the 3D plot too
    if q == 0:
        plt.plot(errors2_poly_extrap[:2000])
        plt.title('Testing Trajectory Error, D2R2, Extrapolation')
        plt.show()

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(ResPred2_poly_extrap[:2000, 0], ResPred2_poly_extrap[:2000, 1], ResPred2_poly_extrap[:2000, 2], 'r', alpha = .5)
        ax.plot(states_extrap[-Ntest:-Ntest + 2000, 0], states_extrap[-Ntest:-Ntest + 2000, 1], states_extrap[-Ntest:-Ntest + 2000, 2], 'b', alpha = .5)
        plt.title('Testing Trajectories, D2R2, Extrapolation', fontsize=16)
        plt.legend(['Predicted', 'True'], fontsize=16, loc=3)
        plt.xlabel('X', fontsize=16)
        plt.ylabel('Y', fontsize=16)
        ax.set_zlabel('Z', fontsize=16)
        plt.draw()
        plt.show()

    # Show errors plot (not just 3D), and also do the Wasserstein metric
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
        projWass[i]  = wasserstein_distance((ResPred2_poly_interp @ proj).flatten(), (states_interp[-Ntest:, :] @ proj).flatten())
        projWass2[i] = wasserstein_distance((ResPred2_poly_extrap @ proj).flatten(), (states_extrap[-Ntest:, :] @ proj).flatten())
    Wasser[q, 0] = np.mean(projWass)
    Wasser[q, 1] = np.mean(projWass2)

# Save data of interest - Wasser

savedat = False
path = '' # Path and filename to store
if savedat:
    with open(path, 'wb') as f:
            pickle.dump([Wasser], f)