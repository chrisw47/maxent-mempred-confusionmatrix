import numpy as np
import pylab as pl

bDe1 = -17.3
bDe2 = -16.2
bDe3 = -13.7
bDe4 = -10.5
prob = []
repressors = np.arange(0,1000) # does not run when set to 10000
NNS = 5000000
for repressor in repressors:
    states = [] #empty, #RNAP Bound, #LacR Bound
    emptyWeight = 1
    RNAPBoundgreenWeight = 1e-3 #(P/NNS)*np.exp(-bDePD)
    LacBoundGreenWeight = 2*repressor*np.exp(-bDe4)/NNS
    totalWeight = emptyWeight+RNAPBoundgreenWeight+LacBoundGreenWeight
    #states.append(emptyWeight/totalWeight)
    #states.append(RNAPBoundgreenWeight/totalWeight)
    #states.append(LacBoundGreenWeight/totalWeight)
    states = [(emptyWeight+LacBoundGreenWeight)/totalWeight,RNAPBoundgreenWeight/totalWeight]
    prob.append(states)

####

prob = np.asarray(prob)

sX, sY = prob.shape

# What is the estimate of the environment given a particular sensor state?

sensor_states = np.arange(sY)
envt_estimates = []
for sensor in sensor_states:
	psGx = prob[:,sensor]
	ind = np.argmax(psGx)
	repressor_guess = repressors[ind]
	envt_estimates.append(repressor_guess)
envt_estimates = np.asarray(envt_estimates)

#envt_estimates = envts[np.argmax(prob,axis=0)]

# Calculate the bias for each environment

bias = []
i = 0
for envt in repressors:
	psGx = prob[i,:]
	bias.append(np.sum(psGx*envt_estimates)-envt)
	i += 1
bias = np.asarray(bias)

#bias = np.dot(prob,envt_estimates)-envts

# Calculate the variance for each environment

variance = []
i = 0
for envt in repressors:
	psGx = prob[i,:]
	avg = np.sum(psGx*envt_estimates)
	avg2 = np.sum(psGx*envt_estimates**2)
	variance.append(avg2-avg**2)
	i += 1
variance = np.asarray(variance)

#variance = np.dot(prob,envt_estimates**2)-np.dot(prob,envt_estimates)**2

# calculate MSE

MSE = bias**2+variance

# Plot them all!

pl.plot(repressors,bias1,'-.',label=r'$\Delta\epsilon_{rd}=-17.3k_BT$')
pl.plot(repressors,bias2,'-.',label=r'$\Delta\epsilon_{rd}=-16.2k_BT$')
pl.plot(repressors,bias3,'-.',label=r'$\Delta\epsilon_{rd}=-13.7k_BT$')
pl.plot(repressors,bias4,'-.',label=r'$\Delta\epsilon_{rd}=-10.5k_BT$')
pl.legend(loc='best')
pl.xscale('log')
pl.xlabel(r'$R$',size=20)
pl.ylabel(r'$Bias(R)$',size=20)
pl.savefig('BiaslacR.pdf',bbox_inches='tight')
pl.show()

pl.plot(repressors,variance1,'-.',label=r'$\Delta\epsilon_{rd}=-17.3k_BT$')
pl.plot(repressors,variance2,'-.',label=r'$\Delta\epsilon_{rd}=-16.2k_BT$')
pl.plot(repressors,variance3,'-.',label=r'$\Delta\epsilon_{rd}=-13.7k_BT$')
pl.plot(repressors,variance4,'-.',label=r'$\Delta\epsilon_{rd}=-10.5k_BT$')
pl.legend(loc='best')
pl.xscale('log')
pl.yscale('log')
pl.xlabel(r'$R$',size=20)
pl.ylabel(r'$Var(R)$',size=20)
pl.savefig('VarlacR.pdf',bbox_inches='tight')
pl.show()

pl.plot(repressors,MSE1,'-.',label=r'$\Delta\epsilon_{rd}=-17.3k_BT$')
pl.plot(repressors,MSE2,'-.',label=r'$\Delta\epsilon_{rd}=-16.2k_BT$')
pl.plot(repressors,MSE3,'-.',label=r'$\Delta\epsilon_{rd}=-13.7k_BT$')
pl.plot(repressors,MSE4,'-.',label=r'$\Delta\epsilon_{rd}=-10.5k_BT$')
pl.legend(loc='best')
pl.xscale('log')
pl.xlabel(r'$R$',size=20)
pl.ylabel(r'$MSE(R)$',size=20)
pl.savefig('MSElacR.pdf',bbox_inches='tight')
pl.show()