import numpy as np
import pylab as pl

xs = np.power(10,np.linspace(-5,7,1000))
pactives = np.asarray([0.001/(1.001+x*np.exp(-(Energies[3]))/5e6) for x in xs])
p = np.vstack([pactives,1-pactives]).T

sX, sY = p.shape

#generate the initial guess
r = np.random.dirichlet(alpha=np.ones(sX));

#Since Csiszar and Tusnady showed that the algorithm converges, we do 
#not need to use Blahut's explicit convergence test, we can just see
#whether the results have stopped changing.
q = np.zeros([sY,sX])
for kk in range(10000):
    #compute the new q
    for x in range(sX):
        for y in range(sY):
            q[y,x] = r[x]*p[x,y]/np.sum(r*p[:,y])

    #compute the new r
    for x in range(sX):
        r[x] = np.exp(np.nansum(p[x,:]*np.log(q[:,x])))
    r /= np.sum(r)


pX = r
#print(pX)
pl.plot(pX); pl.xscale('log'); pl.show()

pY = np.dot(p.T,r)
print(pY)

C = -np.nansum(pY*np.log2(pY))
for ii in range(sX):
    C = C + pX[ii]*np.nansum(p[ii,:]*np.log2(p[ii,:]))

print(C)

###

opt_pcs.append(pX)
opt_pYs.append(pY)
Cs.append(C)

###

pl.rcParams['text.usetex'] = True
pl.plot(xs,opt_pcs[0],label=r'$\Delta\epsilon_{rd}=-17.3k_BT$')
pl.plot(xs,opt_pcs[1],label=r'$\Delta\epsilon_{rd}=-16.2k_BT$')
pl.plot(xs,opt_pcs[2],label=r'$\Delta\epsilon_{rd}=-13.7k_BT$')
pl.plot(xs,opt_pcs[3],label=r'$\Delta\epsilon_{rd}=-10.5k_BT$')
pl.xscale('log')
pl.legend(loc='best')
pl.xlabel('$R$',size=16)
pl.ylabel('$p^*(R)$',size=16)
pl.savefig('popt_generegulation.pdf',bbox_inches='tight')
pl.show()

#######################

xs = np.power(10,np.linspace(-10,-3,1000))
KA = 170e-6
KI = 0.05e-6
theta = 1/5.22
pactives = np.asarray([(1+x/KA)**2/((1+x/KA)**2+theta*(1+x/KI)**2) for x in xs])
p = np.vstack([pactives,1-pactives]).T

sX, sY = p.shape

#generate the initial guess
r = np.random.dirichlet(alpha=np.ones(sX));

#Since Csiszar and Tusnady showed that the algorithm converges, we do 
#not need to use Blahut's explicit convergence test, we can just see
#whether the results have stopped changing.
q = np.zeros([sY,sX])
for kk in range(10000):
    #compute the new q
    for x in range(sX):
        for y in range(sY):
            q[y,x] = r[x]*p[x,y]/np.sum(r*p[:,y])

    #compute the new r
    for x in range(sX):
        r[x] = np.exp(np.nansum(p[x,:]*np.log(q[:,x])))
    r /= np.sum(r)


pX = r
#print(pX)
pl.plot(pX); pl.xscale('log'); pl.show()

pY = np.dot(p.T,r)
print(pY)

C = -np.nansum(pY*np.log2(pY))
for ii in range(sX):
    C = C + pX[ii]*np.nansum(p[ii,:]*np.log2(p[ii,:]))

print(C)

pl.plot(xs,pX)
pl.xscale('log')
pl.legend(loc='best')
pl.xlabel('$c$',size=16)
pl.ylabel('$p^*(c)$',size=16)
pl.savefig('popt_nAChR.pdf',bbox_inches='tight')
pl.show()

###########

xs = np.power(10,np.linspace(-6,-3,1000))
KA1 = 0.5e-3; KA2 = 10**6e-3
KI1 = 0.02e-3; KI2 = 100e-3
theta = np.exp(-1*17.5)
n = 17.5*1; m = 17.5*1.4
pactives = np.asarray([(1+x/KA1)**n*(1+x/KA2)**m/((1+x/KA1)**n*(1+x/KA2)**m+theta*(1+x/KI1)**n*(1+x/KI2)**m) for x in xs])
p = np.vstack([pactives,1-pactives]).T

sX, sY = p.shape

#generate the initial guess
r = np.random.dirichlet(alpha=np.ones(sX));

#Since Csiszar and Tusnady showed that the algorithm converges, we do 
#not need to use Blahut's explicit convergence test, we can just see
#whether the results have stopped changing.
q = np.zeros([sY,sX])
for kk in range(200000):
    #compute the new q
    for x in range(sX):
        for y in range(sY):
            q[y,x] = r[x]*p[x,y]/np.sum(r*p[:,y])

    #compute the new r
    for x in range(sX):
        r[x] = np.exp(np.nansum(p[x,:]*np.log(q[:,x])))
    r /= np.sum(r)


pX = r
#print(pX)

pY = np.dot(p.T,r)
print(pY)

C = -np.nansum(pY*np.log2(pY))
for ii in range(sX):
    C = C + pX[ii]*np.nansum(p[ii,:]*np.log2(p[ii,:]))

print(C)

pl.plot(xs,pX)
pl.xscale('log')
pl.legend(loc='best')
pl.xlabel('$c$',size=16)
pl.ylabel('$p^*(c)$',size=16)
pl.savefig('popt_bacterialchemotaxis.pdf',bbox_inches='tight')
pl.show()
