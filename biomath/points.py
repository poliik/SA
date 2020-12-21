import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
from scipy.optimize import fsolve
from matplotlib.animation import FuncAnimation

def f(u, k=2.):
    return u ** 3. * np.exp(k * (1 - u))

def f_diff(u, z=2.):
    return f(u, z) - u

def fu(u, k=2.):
    return u ** 2. * np.exp(k * (1 - u)) * (3 - k * u)

def g(u, z = 2.):
    return u ** 2. * np.exp(z * (1 - u)) - 1

u = np.linspace(0, 15, 1000)
# z_list = np.linspace(0.3, 5, 14)
z_list = [0.004 * x for x in range(100, 500)]
# z_list = [0.004 * x for x in range(500, 900)]
print(z_list)

fig = plt.figure(figsize=(20, 10))
ax = plt.gca()


def anima(i):
    plt.cla()
    ax.set_xlim([-0.1,  15])
    ax.set_ylim([-0.1,  25])
    z = z_list[i]
    ax.set_xlabel('u')
    ax.set_title('z=' + str(z))
    ax.plot(u, u)

    f1 = f(u, z)
    f2 = f(f1, z)
    f3 = f(f2, z)

    ax.plot(u, f1)
    ax.plot(u, f2)
    ax.plot(u, f3)
    ax.legend(['u', 'f', '$f^2$', '$f^3$'])

    # u = f(u)
    # при z > 2 корень на отрезке [0, 2/z]
    root1 = 1.
    if z > 2:
        root1 = fsolve(g, 1. / z, args=z)
    elif z < 2:
        root1 = fsolve(f_diff, 3./z + 1, args=z)
    plt.scatter([0, 1, root1], [0, 1, root1], c='b');

    f1_max = f(3. / z, z); f2_max = f(f1_max, z); f3_max = f(f2_max, z)
#     ax.set_ylim([-0.1,  max(f1_max, f2_max, f3_max) + 0.5])

    

    
#   
ani = FuncAnimation(fig, anima, frames=len(z_list), interval=5) 
plt.show()