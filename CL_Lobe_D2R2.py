# Code for generating experiments, analysis, and plots used in the Climate Lobes publication
# For the vary-lobe experiment: Use 100% of training data from one lobe of L63, and a variable percentage from the other lobe
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

doWasser = False # For the plots only, try 50 trials with no wasser

np.random.seed(1802)

if doWasser:
    Ntrain = 50000
    Ntest = 100000 # long term climate (wasserstein metric)
    Ntrial = 5 # 5 for wasserstein
else:
    Ntrain = 50000
    Ntest = 2000 # Shorter trajectory predictions for plotting (non-Wasserstein) variant
    Ntrial = 25-24 # 25 for plotting


rho_l = 28.0 # 24.74 is critical, above that chaotic
sigma_l = 10.0
beta_l = 8.0 / 3.0
dim = 3

def f(state, t):
    x, y, z = state  # Unpack the state vector
    return sigma_l * (y - x), x * (rho_l - z) - y, x * y - beta_l * z  # Derivatives

state0 = [1.0, 1.0, 1.0]
#t = np.arange(0.0, (Ntrain+Ntest*Ntrial)/200 + .005, 0.005)
t = np.linspace(0.0, (Ntrain+Ntest*Ntrial)/200 + .005, Ntrain+(Ntest*Ntrial+1))

states = odeint(f, state0, t)

useFE = False # use manual forward euler if True
if (useFE):
    states[0,:] = state0 + .005*np.array(f(state0,0))
    for i in range(Ntrain + Ntest*Ntrial):
        states[i+1,:] = states[i,:] + .005*np.array(f(states[i,:],0))

regularized = np.std(states,0)
statemean = np.mean(states,0)
#states /= np.std(states,0) # scaled
#states -= np.mean(states,0) # BAD FOR ESN, # centered
scale = False
center = False

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

# Get true RK4, FE updates equations
from sympy import symbols
xx, yy, zz = symbols('x y z') #??
dt = .005
rho = rho_l
k1x = sigma_l*(yy-xx)
k1y = (xx *(rho_l-zz)-yy)
k1z = (xx*yy - beta_l*zz)
k2x = sigma_l*((yy+.5*dt*k1y) - (xx+.5*dt*k1x))
k2y = ((xx+.5*dt*k1x) *(rho-(zz+.5*dt*k1z))-(yy+.5*dt*k1y))
k2z = ((xx+.5*dt*k1x)*(yy+.5*dt*k1y) - beta_l*(zz+.5*dt*k1z))
k3x = sigma_l*((yy+.5*dt*k2y) - (xx+.5*dt*k2x))
k3y = ((xx+.5*dt*k2x) *(rho-(zz+.5*dt*k2z))-(yy+.5*dt*k2y))
k3z = ((xx+.5*dt*k2x)*(yy+.5*dt*k2y) - beta_l*(zz+.5*dt*k2z))
k4x = sigma_l*((yy+dt*k3y) - (xx+dt*k3x))
k4y = ((xx+dt*k3x) *(rho-(zz+dt*k3z))-(yy+dt*k3y))
k4z = ((xx+dt*k3x)*(yy+dt*k3y) - beta_l*(zz+dt*k3z))
rkx = xx + dt/6 * (k1x + 2*k2x + 2*k3x + k4x)
rky = yy + dt/6 * (k1y + 2*k2y + 2*k3y + k4y)
rkz = zz + dt/6 * (k1z + 2*k2z + 2*k3z + k4z)
from sympy import expand
expand(rkx)
expand(rky)
expand(rkz)
rkx  = expand(rkx).as_poly()
rky  = expand(rky).as_poly()
rkz  = expand(rkz).as_poly()
rkx.coeffs() # doesn't return 0 entries
#rkx.all_coeffs() # only works with  monomials
rkx.coeff_monomial('x') # works - but have to construct each term via loop e.g. 1 x y z x^2 xy xz...
# have to use the same order as polyfeat, so best to just construct monomial to test directly from polyfeat's combos

# What about forward euler?
fex = xx+ dt*sigma_l*(yy-xx)
fey = yy+ dt*(xx *(rho_l-zz)-yy)
fez = zz+ dt*(xx*yy - beta_l*zz)


order = 4
if order==7: # order will now include all true RK4 terms, FAILS even with some right data
    polysize = 120 # order4 + 21 + 28 + 36
if order==6: # FAILS with no right data, appeared to work with .01 (1%)
    polysize = 84
if order==5: # FAILS with no right data
    polysize = 56
if order==4:
    polysize = 35 # order3 + 15
if order==3:
    polysize = 20 # order2 + 10
if order==2:
    polysize = 10 # 1 + 3 + 6

# Get 'true' Wout from RK4 up to specified order
Wexact = np.zeros((polysize, 3))
counter = 0
Wexact[counter,0] = rkx.coeff_monomial("1")
Wexact[counter,1] = rky.coeff_monomial("1")
Wexact[counter,2] = rkz.coeff_monomial("1")
counter+=1
for i in range(order):
    comb = combinations_with_replacement(np.arange(3), i+1)
    combos = list(comb)
    for j in range(len(combos)):
        struse = ''
        for k in range(i+1):
            if combos[j][k]==0:
                struse+='*x'
            elif combos[j][k]==1:
                struse+='*y'
            else:
                struse+='*z'
        struse = struse[1:]
        Wexact[counter, 0] = rkx.coeff_monomial(struse)
        Wexact[counter, 1] = rky.coeff_monomial(struse)
        Wexact[counter, 2] = rkz.coeff_monomial(struse)
        counter+=1

Wout_poly_store = np.zeros((Ntrial, polysize, 3))
errors2_poly_store = np.zeros((Ntrial, Ntest))
Wasser = np.zeros((Ntrial,4)) # per-dimension wasserstein distances

for q in range(Ntrial):
    print(q)
    # Copy states, and use that local copy only...
    statesuse = states[q*Ntest:Ntrain+q*Ntest + Ntest,:]

    # Partial Training. Can generate other splits as well - e.g. split based on min[d(x,center_1),d(x,center_2)]
    # Attractors: +/- sqrt(Beta*(rho-1)), +/- sqrt(Beta*(rho-1)), rho-1
    # e.g. for our standard numbers, +/- sqrt(72), +/- sqrt(72), 27
    trainper = True
    leftrate = 1.0
    rightrate = 1.0 # 1/.1/.01/.001/0
    if trainper:
        leftdist  = np.sqrt((statesuse[:-Ntest, 0] - np.sqrt(72))**2 + (statesuse[:-Ntest, 1] - np.sqrt(72))**2 + (statesuse[:-Ntest, 2] - 27)**2)
        rightdist = np.sqrt((statesuse[:-Ntest, 0] + np.sqrt(72))**2 + (statesuse[:-Ntest, 1] + np.sqrt(72))**2 + (statesuse[:-Ntest, 2] - 27)**2)
        leftuse = np.where(leftrate* leftdist < rightdist)[0]
        rightuse = np.where(leftrate * leftdist > rightdist)[0]
        totrain = np.concatenate([np.random.permutation(leftuse)[:int(leftrate*len(leftuse))],np.random.permutation(rightuse)[:int(rightrate*len(rightuse))]])
        rightextranum = 25
        rightextra = np.random.permutation(rightuse)[:rightextranum]
        if (center):
            statesuse-= np.mean(statesuse,0)
        if (scale):
            regularize = np.std(statesuse,0)
            statesuse /= regularize

    polyfeatures, order = polyfeat(np.transpose(statesuse[totrain, :]), order)
    pp = PolynomialFeatures(degree=order)
    polyfeattrain = polyfeatures[:, :Ntrain]

    Id_n = np.identity(np.shape(polyfeattrain)[0])
    beta = .0001  # With left only, setting beta ~>1 fails during predicting - blows up, setting to 0 is no better...
    U = np.dot(polyfeattrain, polyfeattrain.transpose()) + Id_n * beta
    Uinv = np.linalg.inv(U)
    Wout_poly = np.dot(Uinv, np.dot(polyfeattrain, statesuse[totrain + 1, :]))
    sigma2 = np.var(polyfeattrain.transpose() @ Wout_poly)


    ResTrain_poly = polyfeattrain.transpose() @ Wout_poly
    errors_og_poly = np.sqrt(np.sum((ResTrain_poly - statesuse[totrain + 1, :]) ** 2, 1))

    # Future predictions - requires both unrolling reservoir (for first 100 Wout) and using polyfeatures (for the remainder)
    polyfeaturestest = np.zeros((Ntest, polyfeatures.shape[0]))  # *2 if using dual exp, exp(-)
    laststate = statesuse[-Ntest-1,:]
    for i in range(Ntest):
        polyfeaturesuse = pp.fit_transform((laststate).reshape(1, -1)).squeeze()
        nextstate = polyfeaturesuse @ Wout_poly
        polyfeaturestest[i, :] = polyfeaturesuse
        laststate = nextstate

    # Generate final output
    ResPred2_poly = polyfeaturestest @ Wout_poly

    if scale:
        ResPred2_poly*=regularized
        statesuse*=regularized


    errors2_poly = np.sqrt(
        np.sum((ResPred2_poly - statesuse[-Ntest:, :]) ** 2, 1))

    Wout_poly_store[q, :, :] = Wout_poly
    errors2_poly_store[q, :] = errors2_poly
    Wasser[q, 0] = wasserstein_distance(ResPred2_poly[:, 0], states[1000:, 0])
    Wasser[q, 1] = wasserstein_distance(ResPred2_poly[:, 1], states[1000:, 1])
    Wasser[q, 2] = wasserstein_distance(ResPred2_poly[:, 2], states[1000:, 2])

    if doWasser: # Instead do what was suggested by Matthew - take average over many random (unit) projections
        numWasser = 100
        projWass = np.zeros(numWasser)
        loadWasser = True
        if loadWasser:
            import pickle
            file_proj = open('.\ESN\Projs.pckl', 'rb')
            test2 = pickle.load(file_proj)
            projs = test2
        for i in range(numWasser):
            proj = projs[:,i]
            proj /= np.linalg.norm(proj)
            projWass[i] = wasserstein_distance((ResPred2_poly @ proj).flatten(), (states[1000:,:] @ proj).flatten())
        Wasser[q,3] = np.mean(projWass)

# Save data of interest - errors2_poly_store, Wasser

savedat = False
path = '' # Path and filename to store
if savedat:
    with open(path, 'wb') as f:
            pickle.dump([errors2_poly_store, Wasser], f)