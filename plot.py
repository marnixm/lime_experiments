import matplotlib
import matplotlib.pyplot as plt


dep = 2
wid = 3
fig, ax = plt.subplots(dep, wid, sharex=True, sharey=True)

for d in range(dep):
    for w in range(wid):
        #ax[d, w].set_title('plot '+ str((d,w)))
        ax[d, w].boxplot([1,2,2,3,4,5,3,3,2,2,4,5,8,5,5,5], positions=[0])
        ax[d, w].boxplot([a+2 for a in [1,2,2,3,4,5,3,3,2,2,4,5,8,5,5,5]], positions=[1])

plt.show()
