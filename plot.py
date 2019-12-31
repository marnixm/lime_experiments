import matplotlib
import matplotlib.pyplot as plt


dep = 2
wid = 2
fig, ax = plt.subplots(dep, wid, sharex=True, sharey=True)

for d in range(dep):
    for w in range(wid):
        ax[d, w].set_title('plot '+ str((d,w)))
        ax[d, w].plot(range(10))

plt.show()
