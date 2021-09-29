# Script to Generate Plots for Climate Lobe Paper

# Data used in this particular script is provided in the Github Repo.
# All data is also reproducible via the other generating scripts.
# Some figures are generated directly by those scripts (e.g. output of a particular example run). This script mostly gives overall plots

import pickle
import matplotlib.pyplplot as plt
import numpy as np

path = '.\ESN\CL Paper\\'

# Figure 1 - ESN Variants
file_poly = open(path+'RegLSRL1R0_25.pckl', 'rb')
test = pickle.load(file_poly)
R1err = test[0]
file_poly = open(path+'RegLSRL1R.001_25.pckl', 'rb')
test = pickle.load(file_poly)
R2err = test[0]
file_poly = open(path+'RegLSRL1R.01_25.pckl', 'rb')
test = pickle.load(file_poly)
R3err = test[0]
file_poly = open(path+'RegLSRL1R.1_25.pckl', 'rb')
test = pickle.load(file_poly)
R4err = test[0]
file_poly = open(path+'RegLSRL1R1_25.pckl', 'rb')
test = pickle.load(file_poly)
R5err = test[0]

plt.plot(np.mean(R1err, 0))
plt.plot(np.mean(R2err, 0))
plt.plot(np.mean(R3err, 0))
plt.plot(np.mean(R4err, 0))
plt.plot(np.mean(R5err, 0))
plt.legend(['0% Right', '.1% Right', '1% Right', '10% Right', '100% Right'], fontsize = 14)
plt.title('Avg Testing Error (Regularized LSR)', fontsize = 16)
plt.ylabel('Error', fontsize = 16)
plt.xlabel('Testing Step', fontsize = 16)
plt.grid(True)
plt.show()
# And 25 HSR
file_poly = open(path+'HSRL1R0_25.pckl', 'rb')
test = pickle.load(file_poly)
R1err = test[0]
file_poly = open(path+'HSRL1R.001_25.pckl', 'rb')
test = pickle.load(file_poly)
R2err = test[0]
file_poly = open(path+'HSRL1R.01_25.pckl', 'rb')
test = pickle.load(file_poly)
R3err = test[0]
file_poly = open(path+'HSRL1R.1_25.pckl', 'rb')
test = pickle.load(file_poly)
R4err = test[0]
file_poly = open(path+'HSRL1R1_25.pckl', 'rb')
test = pickle.load(file_poly)
R5err = test[0]

plt.plot(np.mean(R1err, 0))
plt.plot(np.mean(R2err, 0))
plt.plot(np.mean(R3err, 0))
plt.plot(np.mean(R4err, 0))
plt.plot(np.mean(R5err, 0))
plt.legend(['0% Right', '.1% Right', '1% Right', '10% Right', '100% Right'], fontsize = 14)
plt.title('Avg Testing Error (HSR)', fontsize = 16)
plt.ylabel('Error', fontsize = 16)
plt.xlabel('Testing Step', fontsize = 16)
plt.grid(True)
plt.show()


# Figure 3 (left) - Wasserstein Distance Metric on Vary Lobe Experiment
file_poly = open(path+'VaryLobeWass.pckl', 'rb')
test = pickle.load(file_poly)
D2R2m = test[0]
D2R2sd = test[1]
HSRm = test[2]
HSRsd = test[3]
LSRm = test[4]
LSRsd = test[5]

Rper = np.array([0 + .1, 1, 2, 5, 10, 25, 50, 100])


plt.plot(Rper, D2R2m, label='D2R2')
plt.scatter(Rper, D2R2m, c='b')
plt.plot(Rper, LSRm, label='LSR')
plt.scatter(Rper, LSRm, c='orange')
plt.plot(Rper, HSRm, label='HSR')
plt.scatter(Rper, HSRm, c='g')
plt.semilogx()
plt.semilogy()
plt.legend()
plt.ylim([.1, 11])
plt.xlabel('Percentage of Right Lobe Training Data Used')
plt.ylabel('Wasserstein Distance')
plt.title('Effect of Other Lobe Data on Long Term System Properties')
plt.xticks([.1, 1, 10, 100], ['0', '1', '10', '100'])
plt.grid(True)
plt.show()

# SI - Activation Function Histogram of ESN Variants:

file_poly = open(path+'LSR_base_hist.pckl', 'rb')
test = pickle.load(file_poly)
R0 = test[0]

file_poly = open(path+'LSR_reg_hist.pckl', 'rb')
test = pickle.load(file_poly)
R1 = test[0]

file_poly = open(path+'HSR_base_hist.pckl', 'rb')
test = pickle.load(file_poly)
R2 = test[0]

file_poly = open(path+'HSR_reg_hist.pckl', 'rb')
test = pickle.load(file_poly)
R3 = test[0]

fig, ax = plt.subplots(1, 1)
ax.hist(R0, edgecolor="black")
ax.set_title('Slope of Activation Histogram, LSR, Unregularized', fontsize=14)
ax.set_xlabel('Slope', fontsize=14)
ax.set_ylabel('Count', fontsize=14)
plt.show()

fig, ax = plt.subplots(1, 1)
ax.hist(R1, edgecolor="black")
ax.set_title('Slope of Activation Histogram, LSR, Regularized', fontsize=14)
ax.set_xlabel('Slope', fontsize=14)
ax.set_ylabel('Count', fontsize=14)
plt.show()

fig, ax = plt.subplots(1, 1)
ax.hist(R2, edgecolor="black")
ax.set_title('Slope of Activation Histogram, HSR, Unregularized', fontsize=14)
ax.set_xlabel('Slope', fontsize=14)
ax.set_ylabel('Count', fontsize=14)
plt.show()

fig, ax = plt.subplots(1, 1)
ax.hist(R3, edgecolor="black")
ax.set_title('Slope of Activation Histogram, HSR, Regularized', fontsize=14)
ax.set_xlabel('Slope', fontsize=14)
ax.set_ylabel('Count', fontsize=14)
plt.show()

# Figure 3 (right) - Wasserstein Distance over interpolation/extrapolation of varying rho

xticks = np.array([8, 10, 16, 20, 23.5, 25, 26, 28, 30, 34])

file_poly = open(path+'VaryRhoWass.pckl', 'rb')
test = pickle.load(file_poly)
HSR = test[0]
LSR = test[1]
D2R2 = test[2]


plt.semilogy(xticks, HSR)
plt.semilogy(xticks, LSR)
plt.semilogy(xticks, D2R2)
plt.scatter(xticks, HSR)
plt.scatter(xticks, LSR)
plt.scatter(xticks, D2R2)
# plt.hlines([12,24])
plt.axvspan(12, 24, color='yellow', alpha=0.15)
plt.axvspan(24, 35, color='blue', alpha=.1)
plt.axvspan(6, 12, color='blue', alpha=.1)
plt.text(15, 4e-3, r'Interpolation Region')
plt.text(25, 4e-3, r'Extrapolation Region')
plt.legend(['HSR', 'LSR', 'D2R2'], fontsize=14)
plt.xlabel('$\\rho$ Value', fontsize=14)
plt.ylabel('Wasserstein Distance', fontsize=14)
plt.title('Interpolation and Extrapolation over Varying $\\rho$')
plt.grid(True)
plt.xlim([8, 34])
plt.show()

# SI - Scaled vs Unscaled D2R2

# Load and plot scaled vs not
file_poly = open(path+'L1R1.pckl', 'rb')
test = pickle.load(file_poly)
RWout = test[0]
Rerr = test[1]
file_poly = open(path+'L1R1Scaled.pckl', 'rb')
test = pickle.load(file_poly)
RSWout = test[0]
RSerr = test[1]
plt.plot(np.mean(Rerr, 0))
plt.plot(np.mean(RSerr[:, :2000], 0))
plt.legend(['D2R2', 'Scaled D2R2'], fontsize=14)
plt.title('Avg Testing Error', fontsize=16)
plt.xlabel('Testing Step', fontsize=16)
plt.ylabel('Error', fontsize=16)
plt.grid(True)
plt.show()

# Fig 1: D2R2
# Plot 25 version
file_poly = open(path+'L1R0_25.pckl', 'rb')
test = pickle.load(file_poly)
R1Wout = test[0]
R1err = test[1]
file_poly = open(path+'L1R.001_25.pckl', 'rb')
test = pickle.load(file_poly)
R2Wout = test[0]
R2err = test[1]
file_poly = open(path+'L1R.01_25.pckl', 'rb')
test = pickle.load(file_poly)
R3Wout = test[0]
R3err = test[1]
file_poly = open(path+'L1R.1_25.pckl', 'rb')
test = pickle.load(file_poly)
R4Wout = test[0]
R4err = test[1]
file_poly = open(path+'L1R1_25.pckl', 'rb')
test = pickle.load(file_poly)
R5Wout = test[0]
R5err = test[1]

plt.plot(np.mean(R1err, 0))
plt.plot(np.mean(R2err, 0))
plt.plot(np.mean(R3err, 0))
plt.plot(np.mean(R4err, 0))
plt.plot(np.mean(R5err, 0))
plt.legend(['0% Right', '.1% Right', '1% Right', '10% Right', '100% Right'], fontsize=14)
plt.title('Avg Testing Error (D2R2)', fontsize=16)
plt.ylabel('Error', fontsize=16)
plt.xlabel('Testing Step', fontsize=16)
plt.grid(True)
plt.show()