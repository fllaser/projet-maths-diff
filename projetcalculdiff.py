# Third-Party Libraries
# ---------------------

# Autograd & Numpy
import autograd
import autograd.numpy as np

# Pandas
import pandas as pd

# Matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 10] # [width, height] (inches). 

# Jupyter & IPython
from IPython.display import display



def grad(f):
    g = autograd.grad
    def grad_f(x, y):
        return np.array([g(f, 0)(x, y), g(f, 1)(x, y)])
    return grad_f


def J(f):
    j = autograd.jacobian
    def J_f(x, y):
        return np.array([j(f, 0)(x, y), j(f, 1)(x, y)]).T
    return J_f


def f(x, y):
    return np.sin(x) + 2.0 * np.sin(y)

grad_f = grad(f)

grad_f(0.0, 0.0) # array([1., 2.])

def f(x, y):
    return np.array([x + 2 * y, 3 * x + 4 * y])

J_f = J(f)

J_f(0.0, 0.0) # array([[1., 2.], [3., 4.]])

def display_contour(f, x, y, levels):
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    fig, ax = plt.subplots()
    contour_set = plt.contour(
        X, Y, Z, colors="grey", linestyles="dashed", 
        levels=levels 
    )
    ax.clabel(contour_set)
    plt.grid(True)
    plt.xlabel("$x_1$") 
    plt.ylabel("$x_2$")
    plt.gca().set_aspect("equal")

def f1(x1, x2):
    x1 = np.array(x1)
    x2 = np.array(x2)
    return 3.0 * x1 * x1 - 2.0 * x1 * x2 + 3.0 * x2 * x2 

def f2(x1, x2):
    return (x1 - 1)**2 + (x1 - x2**2)**2


N=100
eps=1e-10

Pointsx=[]
Pointsy=[]


def Newton(F, x0, y0, eps=eps, N=N):
    Jacf=J(F)
    Pointsx.append(x0)
    Pointsy.append(y0)
    for _ in range(N):
        Z=np.array([x0,y0])
        FZ=np.array(F(x0,y0))
        Jinv=np.linalg.inv(Jacf(x0,y0))
        Z = Z - np.transpose(np.dot(Jinv, np.transpose(FZ) ))
        x = Z[0]
        y = Z[1]
        Pointsx.append(x)
        Pointsy.append(y)
        if np.sqrt((x - x0)**2 + (y - y0)**2) <= eps:
            return x, y
        x0, y0 = x, y
    else:
        raise ValueError(f"no convergence in {N} steps.")



c=0.8

def G(x,y):
    return np.array([f1(x,y)-c,x-y])


N=Newton(G,c,c)
print(N,f1(N[0],N[1]))


display_contour(
    f1, 
    x=np.linspace(-1.0, 1.0, 100), 
    y=np.linspace(-1.0, 1.0, 100), 
    levels=10 # 10 levels, automatically selected
)
plt.plot(Pointsx,Pointsy,'bo')
i=1
for x, y in zip(Pointsx, Pointsy):
    plt.text(x, y, str(i), color="red", fontsize=12)
    i+=1
plt.plot([-1,1],[-1,1],'--')
#plt.show()




def intersectiondroite(f,x0,y0,delta):
    grad_f = grad(f)
    g=grad_f(x0, y0)
    new_norme=delta/np.sqrt(g[0]**2+g[1]**2)
    ortho=[g[1]*new_norme,-g[0]*new_norme]
    return (x0+ortho[0],y0+ortho[1])


#interctiondroite(f1,0.6,0.6,2)

def level_curve(f, x0, y0, delta=0.1, N=40, eps=eps):
    res=np.zeros((2,N))
    c=f(x0,y0)
    for i in range (N):
        def T3(x,y):
            return np.array([f(x,y)-c,(x0-x)**2+(y0-y)**2-delta**2])
        x1,y1=intersectiondroite(f,x0,y0,delta)
        x2,y2=Newton(T3,x1,y1,eps)
        res[0][i],res[1][i]=x2,y2
        x0,y0=x2,y2
    return res

LC=level_curve(f1, 0.4,0.7)

print(LC,c)
plt.plot(LC[0],LC[1],'bo',color='green')
i=1
for x, y in zip(LC[0], LC[1]):
    plt.text(x, y, str(i), color="green", fontsize=12)
    i+=1
plt.show()