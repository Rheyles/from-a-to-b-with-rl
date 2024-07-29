import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

r = np.linspace(0,2,50)
clipped = np.maximum(-r, -1.2)

ax.plot(r,-r, "k--")
ax.plot(r, -1.2*np.ones_like(r), "k--")
ax.plot(r, clipped, 'r')

ax.plot([0.8,0.8],[-0.8,0], ':', color='gray')
ax.plot([1.0,1.0],[-1.0,0], ':', color='lightgray')
ax.plot([1.2,1.2],[-1.2,0], ":", color='gray')


ax.annotate("$1 + \\epsilon$", xy=(1.13, 0.1))
ax.annotate("$1 - \\epsilon$", xy=(0.73, 0.1))
ax.annotate("$1$", xy=(0.97, 0.1))
ax.annotate("$\\Upsilon (s,a)$", xy=(1.9, 0.1))
ax.annotate("Loss", xy=(0.05, 0.5))

ax.annotate("", xy=(2.1, 0), xytext=(0, 0),
             arrowprops=dict(arrowstyle="->"))
ax.annotate("", xy=(0, 0.5), xytext=(0,-2.1),
             arrowprops=dict(arrowstyle="->"))

ax.set_ylabel('Loss')
ax.set_xticks([0,0.8,1,1.2])
ax.set_yticks([0])
ax.set_yticklabels(['0'])
ax.set_xticklabels(['0', '$1-\\epsilon$', '1', '$1 + \\epsilon$'])
ax.set_title("Case where $A > 0$")
ax.set_xlabel("$\\Upsilon (s,a)$")

plt.axis('off')
ax.axis([-0.1,2.1,-1.5,0.5])

plt.show()
fig.savefig('toto.svg')