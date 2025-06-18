import numpy as np
from scipy.integrate import dblquad
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

title_font = {'fontname':'Arial', 'size':'20', 'color':'black', 'weight':'normal', 'verticalalignment':'bottom'} # Bottom vertical alignment for more space 
axis_font = {'fontname':'Arial', 'size':'14'}


def f_exp(y1,lam_exp):
    'Función de densidad distribución exponencial'
    expo=np.exp(-y1*lam_exp)*lam_exp #densidad exponencial.
    return(expo)

def F_exp(y1,lam_exp):
    'Función de distribución exponencial'
    F=1-np.exp(-y1*lam_exp)
    return(F)

def f_unif(y2,a,b):
    'Función de densidad uniforme'
    return np.where(y2>=a,np.where(y2<=b,1/(b-a),0),0)

def F_unif(y2,a,b):
    'Función de distribuación uniforme'
    return np.where(y2<a,0,np.where(y2<=b,(y2-a)/(b-a),1))

def cuantil_unif(p2,a,b):
    'Función cuantil (inversa de la distribución uniforme)'
    return(b*p2+a*(1-p2))

def cuantil_exp(p1,lam_exp):
    'Función cuantil exponencial'
    return(-np.log(1-p1)/lam_exp)

def funcion_Q_M_2(mu,sig,p,r,eta,theta,x1,x2,L1,L2,lam1,lam2,lam_exp,a_unif,b_unif):
    "integrando 3.8, con y1 sim exponencial y y2 sim uniforme"
    return (np.exp(r*eta*(cuantil_exp(1-x1,lam_exp)+cuantil_exp(1-x2,lam_exp)))-1)*L1*L2*lam1*lam2*(1+theta)*(((L1*lam1*x1)**(-theta)+(L2*lam2*(x2))**(-theta))**(-((1/theta)+2)))*((L1*L2*lam1*lam2*(x1)*(x2))**(-(theta+1)))

def funcion_Q_L(mu,sig,p,r,eta,theta,L1,L2,lam1,lam2,lam_exp,a_unif,b_unif):
    "Q case Arquimedia"
    "funcion Q(L)"
    return (1/2*((mu-r)/sig)**2+r*eta*(L1*p[0]+L2*p[1])-dblquad(lambda x1,x2: funcion_Q_M_2(mu,sig,p,r,eta,theta,x1,x2,L1,L2,lam1,lam2,lam_exp,a_unif,b_unif),0.0001,1,0.0001,1)[0])

def integrando_Q(r,eta,x1,x2,L1,L2,lam1,lam2,lam_exp,a_unif,b_unif):
    'Integrando copula S. Ismail 3.6'
    return (np.exp(r*eta*(cuantil_exp(1-x1,lam_exp)+cuantil_unif(1-x2,a_unif,b_unif)))-1)*((L1*L2*lam1*lam2)**2)*2*(x1*x2)/((L1*lam1*x1+L2*lam2*x2-L1*L2*lam1*lam2*x1*x2)**3)

def funcion_Q_L(mu,sig,p,r,eta,L1,L2,lam1,lam2,lam_exp,a_unif,b_unif):
    "funcion Q(L)"
    aux = dblquad(lambda x1,x2: integrando_Q(r,eta,x1,x2,L1,L2,lam1,lam2,lam_exp,a_unif,b_unif),0.00005,1,0.00005,1)
    return (1/2*((mu-r)/sig)**2+r*eta*(L1*p[0]+L2*p[1])-aux[0])

def fun_alpha_j(N,beta_j,lamb_0_j,p_j):
    "retorna alpha_j"
    return N*np.exp((-beta_j/lamb_0_j)*p_j)*(lamb_0_j+beta_j*p_j)

def integrando_Q_M_2_prima(mu,sig,r,eta,theta,x1,x2,lam_exp1,lam_exp2,b_unif,N1,N2,beta_j_vec,lamb_0_vec,prim_1,prim_2):
    "integrando 2.7, con y1 sim exponencial y y2 sim exponencial"
    alpha_1 = fun_alpha_j(N1,beta_j_vec[0],lamb_0_vec[0],prim_1)
    alpha_2 = fun_alpha_j(N2,beta_j_vec[1],lamb_0_vec[1],prim_2)
    return alpha_1*alpha_2*(np.exp(r*eta*(cuantil_exp(1-x1,lam_exp1)+cuantil_exp(1-x2,lam_exp2)))-1)*(1+theta)*(((alpha_1*x1)**(-theta)+(alpha_2*x2)**(-theta))**(-((1/theta)+2)))*((alpha_1*alpha_2*x1*x2)**(-(theta+1)))

def funcion_Q_prima(mu,sig,r,eta,theta,N1,N2,beta_j_vec,lamb_0_j_vec,prim_vec,lam_exp1,lam_exp2,b_unif):
    "funcion Q(L)"
    aux = dblquad(lambda x1,x2: integrando_Q_M_2_prima(mu,sig,r,eta,theta,x1,x2,lam_exp1,lam_exp2,b_unif,N1,N2,beta_j_vec,lamb_0_j_vec,prim_vec[0],prim_vec[1]),0.00005,1,0.00005,1)
    return (1/2*((mu-r)/sig)**2+r*eta*(N1*np.exp(-beta_j_vec[0]/lamb_0_j_vec[0]*prim_vec[0])*prim_vec[0]+N2*np.exp(-beta_j_vec[1]/lamb_0_j_vec[1]*prim_vec[1])*prim_vec[1])-aux[0])

### Parameters

y1_ej = 5
y2_ej = 15
lam_exp_ej1 = 0.5 #gamma_1
lam_exp_ej2 = 0.7 #gamma_2
a_ej = 1 #parametros uniform
b_ej = 2 #parametros uniform
theta_ej = 5
lam1_ej = 0.1
lam2_ej = 0.05
L1_ej = 3
L2_ej = 2
p1= 0.4
p2= 0.5
p_ej = np.array([p1,p2])

r_ej = 0.04
mu_ej = 0.06
sig_ej = 0.21
eta_ej = 5

#### integrando de Q para primas

beta_1_ej = 0.15
beta_2_ej = 0.24
N1_ej = 100
N2_ej = 80
lam_0_1_ej = 0.2
lam_0_2_ej = 0.7


y1_plot = np.linspace(0,120,200)
y2_plot = np.linspace(0,200,200)


l1 = np.linspace(0.01,10,10) 
l2 = np.linspace(0.01,10,10)
L_1, L_2 = np.meshgrid(l1,l2)

Q_plot = np.empty((len(l2),len(l1)))
for i in np.arange(0,len(l2)):
    for j in np.arange(0,len(l1)):
        Q_plot[i,j] = funcion_Q_L(mu_ej,sig_ej,
                                  p_ej,r_ej,eta_ej,
                                  theta_ej,l1[j],
                                  l2[i],lam1_ej,
                                  lam2_ej,lam_exp_ej1,
                                  a_ej,b_ej)

## Q Surface

figura = plt.figure(figsize=(15.1,14))
ax = plt.axes(projection="3d")
ax.plot_surface(L_1,L_2,Q_plot,cmap="viridis")
ax.set_xlabel('L_1')
ax.set_ylabel('L_2')
ax.set_title(r'Q(L)',**title_font)
plt.show()

## Integrando Surface

l1_plot = np.linspace(0.001,10,100)
l2_plot = np.linspace(0.001,10,100)
L_1, L_2 = np.meshgrid(l1_plot,l2_plot)
Z_plot = integrando_Q(r_ej,eta_ej,0.3,0.5,L_1,L_2,lam1_ej,lam2_ej,lam_exp_ej1,a_ej,b_ej)

figura = plt.figure(figsize=(15.1,14))
ax = plt.axes(projection="3d")
ax.plot_surface(L_1,L_2,Z_plot,cmap="viridis")
ax.set_xlabel('l_1')
ax.set_ylabel('l_2')
ax.set_title('Integrando Q(L), para x1=x2=0.5')
plt.show()


## Q_L plot

l1 = np.linspace(0.01,10,41)
l2 = np.linspace(0.01,10,31)
L_1, L_2 = np.meshgrid(l1,l2)

Q_plot = np.empty((len(l2),len(l1)))

for i in np.arange(0,len(l2)):
    for j in np.arange(0,len(l1)):
        Q_plot[i,j] = funcion_Q_L(mu_ej,sig_ej,
                                  p_ej,r_ej,
                                  eta_ej,l1[j],
                                  l2[i],lam1_ej,
                                  lam2_ej,lam_exp_ej1,
                                  a_ej,b_ej)

figura = plt.figure(figsize=(15.1,14))
ax = plt.axes(projection="3d")
ax.plot_surface(L_1,L_2,Q_plot,cmap="viridis")
ax.set_xlabel('L_1')
ax.set_ylabel('L_2')
ax.set_title(r'Q(L)',**title_font)
plt.show()


## integrando de Q para primas

prim_1_ej = 0.05
prim_2_ej = 0.03

prim1_plot = np.linspace(0.1,10,35)
prim2_plot = np.linspace(0.1,20,35)
Prim_1, Prim_2 = np.meshgrid(prim1_plot,prim2_plot)
Z_plot = integrando_Q_M_2_prima(mu_ej,sig_ej,r_ej,
                                eta_ej,theta_ej,
                                0.2,0.1,lam_exp_ej1,
                                lam_exp_ej2,b_ej,
                                N1_ej,N2_ej,
                                np.array([beta_1_ej,beta_2_ej]),
                                np.array([lam_0_1_ej,lam_0_2_ej]),
                                Prim_1,Prim_2)

# integrando en funcion de las primas

figura = plt.figure(figsize=(15.1,14))
ax = plt.axes(projection="3d")
ax.plot_surface(Prim_1,Prim_2,Z_plot,cmap="viridis")
ax.set_xlabel(r'$prim_1$')
ax.set_ylabel(r'$prim_2$')
ax.set_title('Integrando 2.7')
plt.show()

# integrando en funcion de x.

theta_ej = 0.2
x_1_plot = np.linspace(0.001,0.99,10)
x_2_plot = np.linspace(0.001,0.99,10)
X_1 , X_2 = np.meshgrid(x_1_plot,x_2_plot)
Z_plot_xs = integrando_Q_M_2_prima(mu_ej,sig_ej,
                                   r_ej,eta_ej,theta_ej,
                                   X_1,X_2,lam_exp_ej1,
                                   lam_exp_ej2,b_ej,N1_ej,
                                   N2_ej,np.array([beta_1_ej,beta_2_ej]),
                                   np.array([lam_0_1_ej,lam_0_2_ej]),100,80)

figura = plt.figure(figsize=(5.1,6))
ax = plt.axes(projection="3d")
ax.plot_surface(X_1,X_2,Z_plot_xs,cmap="viridis")
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
ax.set_title('Integrando 2.7')
plt.show()

## Para primas

Q_plot = np.empty((len(prim2_plot),len(prim1_plot)))

for i in np.arange(0,len(prim2_plot)):
    for j in np.arange(0,len(prim1_plot)):
        Q_plot[i,j] = funcion_Q_prima(mu_ej, sig_ej,
                                      r_ej, eta_ej,
                                      theta_ej, N1_ej,
                                      N2_ej, np.array([beta_1_ej,beta_2_ej]),
                                      np.array([lam_0_1_ej,lam_0_2_ej]),
                                      np.array([prim1_plot[j],prim2_plot[i]]),
                                      lam_exp_ej1, lam_exp_ej2, b_ej)

figura = plt.figure(figsize=(15.1,14))
ax = plt.axes(projection="3d")
ax.plot_surface(Prim_1,Prim_2,Q_plot,cmap="viridis")
ax.set_xlabel('p_1', **axis_font)
ax.set_ylabel('p_2', **axis_font)
ax.set_title(r'Q(L)',**title_font)
plt.show()


# Simplex

### Simplex

distan = 1
f_refl = 1
f_expa = 2
f_cont = 0.5
f_rempl =0.5

x0 = np.array([0,0])
x1 = distan/(2*np.sqrt(2))*np.array([np.sqrt(3)+1,np.sqrt(3)-1])
x2 = distan/(2*np.sqrt(2))*np.array([np.sqrt(3)-1,np.sqrt(3)+1])

TOL = 0.00001

puntos = np.array([x0,x1,x2])
Q_val = [funcion_Q_prima(mu_ej, sig_ej, r_ej, eta_ej, theta_ej,
                         N1_ej, N2_ej, np.array([beta_1_ej,beta_2_ej]),
                         np.array([lam_0_1_ej,lam_0_2_ej]), i, lam_exp_ej1,
                         lam_exp_ej2, b_ej) for i in puntos]



while abs(max(Q_val)-min(Q_val)) > TOL:
    centroide = sum(np.delete(puntos,np.argmin(Q_val), axis = 0))/2
    Q_max = max(Q_val)
    Q_min = min(Q_val)
    x_max = puntos[np.argmax(Q_val)]
    x_min = puntos[np.argmin(Q_val)]
    x_new = (1+f_refl)*centroide-f_refl*x_min
    Q_new = funcion_Q_prima(mu_ej, sig_ej, r_ej,
                            eta_ej, theta_ej, N1_ej, N2_ej,
                            np.array([beta_1_ej,beta_2_ej]),
                            np.array([lam_0_1_ej,lam_0_2_ej]),
                            x_new, lam_exp_ej1, lam_exp_ej2, b_ej)
    print(Q_max)
    if Q_max < Q_new:
        x_expa = f_expa*x_new+(1-f_expa)*centroide
        Q_expa = funcion_Q_prima(mu_ej, sig_ej, r_ej,
                                 eta_ej, theta_ej, N1_ej, N2_ej,
                                 np.array([beta_1_ej,beta_2_ej]),
                                 np.array([lam_0_1_ej,lam_0_2_ej]),
                                 x_expa, lam_exp_ej1, lam_exp_ej2, b_ej)
        if Q_expa > Q_new:
           puntos[np.argmin(Q_val)] = x_expa
           Q_val[np.argmin(Q_val)] = Q_expa
        else:
            puntos[np.argmin(Q_val)] = x_new
            Q_val[np.argmin(Q_val)] = Q_new
    elif Q_new > np.delete(Q_val,[np.argmin(Q_val),np.argmax(Q_val)],axis =0):
        puntos[np.argmin(Q_val)] = x_new
        Q_val[np.argmin(Q_val)] = Q_new
    elif Q_new > Q_min:
        x_contra = f_cont*x_min+(1-f_cont)*centroide
        Q_contra = funcion_Q_prima(mu_ej, sig_ej, r_ej, eta_ej,
                                   theta_ej, N1_ej, N2_ej,
                                   np.array([beta_1_ej,beta_2_ej]),
                                   np.array([lam_0_1_ej,lam_0_2_ej]),
                                   x_contra, lam_exp_ej1, lam_exp_ej2, b_ej)
        if Q_contra > Q_min:
            puntos[np.argmin(Q_val)] = x_contra
            Q_val[np.argmin(Q_val)] = Q_contra
        else:
            puntos = np.array([x_max,(1-f_rempl)*x_max+f_rempl*x_min,
                               (1-f_rempl)*x_max+f_rempl*np.delete(puntos,
                                                                   [np.argmax(Q_val),
                                                                    np.argmin(Q_val)],
                                                                   axis=0)[0]])
            Q_val = [funcion_Q_prima(mu_ej, sig_ej, r_ej, eta_ej,
                                     theta_ej, N1_ej, N2_ej,
                                     np.array([beta_1_ej,beta_2_ej]),
                                     np.array([lam_0_1_ej,lam_0_2_ej]),
                                     i, lam_exp_ej1, lam_exp_ej2,
                                     b_ej) for i in puntos]
    else:
        puntos = np.array([x_max,(1-f_rempl)*x_max+f_rempl*x_min,
                           (1-f_rempl)*x_max+f_rempl*np.delete(puntos,
                                                               [np.argmax(Q_val),
                                                                np.argmin(Q_val)],
                                                               axis=0)[0]])
        Q_val = [funcion_Q_prima(mu_ej, sig_ej, r_ej, eta_ej,
                                 theta_ej, N1_ej, N2_ej,
                                 np.array([beta_1_ej,beta_2_ej]),
                                 np.array([lam_0_1_ej,lam_0_2_ej]),
                                 i, lam_exp_ej1, lam_exp_ej2,
                                 b_ej) for i in puntos]

print('El max de Q está en', x_max, r'y $Q^*$ es', Q_max)


contours = plt.contour(Prim_1,Prim_2,Q_plot,
                       levels = [10,20,30,40,47,49], colors='black')
plt.clabel(contours, inline=True, fontsize=8, fmt='%d')
plt.xlabel(r'$n_1$',**axis_font)
plt.ylabel(r'$n_2$',**axis_font)
plt.grid()
plt.show()
