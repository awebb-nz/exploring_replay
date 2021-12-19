import numpy as np
import pandas as pd

gamma   = 0.9
T       = 100
step    = 0.0001
R       = np.zeros((T, T))
gittins = np.zeros_like(R)

for a in range(1, T):
    R[a, T-a] = a/T

for p in np.arange(step, 1, step/2):
    safe = p/(1-gamma)
    for t in reversed(range(1, T)):
        for a in range(1, t):
            risky = (a/t)*(1+gamma*R[a+1, t-a]) + ((t-a)/t) *(gamma*R[a, t-a+1])

            if (gittins[a, t-a] == 0) and (safe>=risky):
                gittins[a, t-a] = p-step/2

            R[a, t-a] = max(safe, risky)

df = pd.DataFrame(gittins.T)
# df.drop(0, axis=1, inplace=True)
# df.drop(0, axis=0, inplace=True)
df.to_csv('gittins%u.csv'%T, index=False)