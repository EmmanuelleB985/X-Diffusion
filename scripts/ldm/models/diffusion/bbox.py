import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

import numpy as np


fig, ax = plt.subplots()

n = 1000
x = np.linspace(-6.3, 6.3, n)
y = np.sin(x) +  0.1*np.random.rand(n)
  
ax.plot(x, y)

axins = zoomed_inset_axes(ax, 6, loc=1) # zoom = 6
axins.plot(x, y)
axins.set_xlim(0, 0.2) # Limit the region for zoom
axins.set_ylim(0, 0.2)
for axis in ['top','bottom','left','right']:
    axins.spines[axis].set_linewidth(1)
    axins.spines[axis].set_color('r')

plt.xticks(visible=False)  # Not present ticks
plt.yticks(visible=False)
#
## draw a bbox of the region of the inset axes in the parent axes and
## connecting lines between the bbox and the inset axes area
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="r")
plt.draw()
plt.show()
plt.savefig('/work/emmanuelle/zero123/zero123/ldm/models/diffusion/test.png')