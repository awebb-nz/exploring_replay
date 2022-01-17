import numpy as np

def gittins_idcs(gamma, T):
    step    = 0.0001
    R       = np.zeros((T, T))
    gittins = np.zeros_like(R)

    for a in range(1, T):
        R[a, T-a] = a/T

    for p in np.arange(step, 1, step/2):
        safe = p/(1-gamma)
        for t in reversed(range(1, T)):
            for a in range(1, t):
                risky = (a/t)*(1+gamma*R[a+1, t-a]) + ((t-a)/t)*(0 + gamma*R[a, t-a+1])

                if (gittins[a, t-a] == 0) and (safe>=risky):
                    gittins[a, t-a] = p-step/2

                R[a, t-a] = max(safe, risky)
                
    return gittins

if __name__ == '__main__':
    gamma = 0.9
    for T in [20, 50, 100, 200]:
        gt = gittins_idcs(gamma, T)
        np.savetxt('gittins%u_gamma09.csv'%T, gt, delimiter=',')
        print('Done with %u'%T)