import numpy as np
from scipy.optimize import minimize
from time import time
import pynof 

def orbopt_sd(gamma,C,H,I,b_mnl,p):
     '''  IN: 
          gamma: Occupation Numbers (ONs)
          C: NOs guess
          H: Monoelectronic Integrals
          I: Repulsion Integrals
          b_mnl: RI approximation Integrals
          p: molecule properties
       OUT:
          best_E: Lowest Energy
          best_C: Optimized NOs 
          nit: number of iterations
          success: Truee if convergence reached
    '''
     
     print("Starting SD ... with alpha",p.alpha)
     ### We calculate the occupations numbers of the Orbitals
     n,dn_dgamma = pynof.ocupacion(gamma,p.no1,p.ndoc,p.nalpha,p.nv,p.nbf5,p.ndns,p.ncwo,p.HighSpin,p.occ_method)
     ## NOF Coefficients, realted with the selected functional
     cj12,ck12 = pynof.PNOFi_selector(n,p)
     y = np.zeros((p.nvar))
     # We calculate the Energy
     E = pynof.calcorbe(y, n,cj12,ck12,C,H,I,b_mnl,p)
     ### Steepest descent variables ### Steep Size.
     step_size=p.alpha

     improved = False
     success = False
     #E is the energy
     #C are the coefficients of the orbitals we want to optimize
     best_E, best_C = E, C
     
     nit = 0
      
     for i in range(p.maxloop):
          nit += 1

          # we calculate the orbital gradient
          grad = pynof.calcorbg(y*0, n,cj12,ck12,C,H,I,b_mnl,p)
          # Check convergence
          if np.linalg.norm(grad) < 10**-4 and improved:
              success = True
              break
              
          # Steepest descent step. Calculate the step size for this variable
          y = - step_size * grad
          
          # Get the new coefficients of the orbitals based on the step = y, the Old coefficients (C), and the properties of our system (p)
          C = pynof.rotate_orbital(y,C,p)
          # Calculate the new energy
          E = pynof.calcorbe(y*0, n,cj12,ck12,C,H,I,b_mnl,p)
 
          # Compare if the energy is better than the previos one.
          if E < best_E:
              best_C = C
              best_E = E
              improved = True

     return best_E,best_C,nit,success


def orbopt_adam2(gamma,C,H,I,b_mnl,p):
    '''  IN: 
          gamma: Occupation Numbers (ONs)
          C: NOs guess
          H: Monoelectronic Integrals
          I: Repulsion Integrals
          b_mnl: RI approximation Integrals
          p: molecule properties
       OUT:
          best_E: Lowest Energy
          best_C: Optimized NOs 
          nit: number of iterations
          success: Truee if convergence reached
    '''
    
    
    if not improved:
        p.alpha = p.alpha/10
        p.maxloop = p.maxloop + 30
        #print("      alpha ",p.alpha)

    return best_E,best_C,nit,success


def orbopt_adam(gamma,C,H,I,b_mnl,p):
    '''  IN: 
          gamma: Occupation Numbers (ONs)
          C: NOs guess
          H: Monoelectronic Integrals
          I: Repulsion Integrals
          b_mnl: RI approximation Integrals
          p: molecule properties
       OUT:
          best_E: Lowest Energy
          best_C: Optimized NOs 
          nit: number of iterations
          success: Truee if convergence reached
    '''
    

    return best_E,best_C,nit,success



def orbopt_cg(gamma,C,H,I,b_mnl,p):
     '''  IN: 
          gamma: Occupation Numbers (ONs)
          C: NOs guess
          H: Monoelectronic Integrals
          I: Repulsion Integrals
          b_mnl: RI approximation Integrals
          p: molecule properties
       OUT:
          best_E: Lowest Energy
          best_C: Optimized NOs 
          nit: number of iterations
          success: Truee if convergence reached
    '''



    return best_E,best_C,nit,success

def orbopt_adagrad(gamma,C,H,I,b_mnl,p):
     '''  IN: 
          gamma: Occupation Numbers (ONs)
          C: NOs guess
          H: Monoelectronic Integrals
          I: Repulsion Integrals
          b_mnl: RI approximation Integrals
          p: molecule properties
       OUT:
          best_E: Lowest Energy
          best_C: Optimized NOs 
          nit: number of iterations
          success: Truee if convergence reached
    '''

    return best_E,best_C,nit,success


def orbopt_rmsprop(gamma,C,H,I,b_mnl,p):
    '''  IN: 
          gamma: Occupation Numbers (ONs)
          C: NOs guess
          H: Monoelectronic Integrals
          I: Repulsion Integrals
          b_mnl: RI approximation Integrals
          p: molecule properties
       OUT:
          best_E: Lowest Energy
          best_C: Optimized NOs 
          nit: number of iterations
          success: Truee if convergence reached
    '''
   

    return best_E,best_C,nit,success


def orbopt_adadelta(gamma, C, H, I, b_mnl, p,):
    '''  IN: 
          gamma: Occupation Numbers (ONs)
          C: NOs guess
          H: Monoelectronic Integrals
          I: Repulsion Integrals
          b_mnl: RI approximation Integrals
          p: molecule properties
       OUT:
          best_E: Lowest Energy
          best_C: Optimized NOs 
          nit: number of iterations
          success: Truee if convergence reached
    '''
    
    return best_E, best_C, nit, success


def comb(gamma,C,H,I,b_mnl,p):

    x = np.zeros((p.nvar+p.nv))
    x[p.nvar:] = gamma

    if("trust" in p.orbital_optimizer or "Newton-CG" in p.orbital_optimizer):
        res = minimize(pynof.calccombeg, x, args=(C,H,I,b_mnl,p),jac=True,hess="2-point",method=p.combined_optimizer,options={"maxiter":p.maxloop})
    else:
        res = minimize(pynof.calccombeg, x, args=(C,H,I,b_mnl,p),jac=True,method=p.combined_optimizer,options={"maxiter":p.maxloop})

    E = res.fun
    x = res.x
    grad = res.jac
    nit = res.nit
    y = x[:p.nvar]
    gamma = x[p.nvar:]
    C = pynof.rotate_orbital(y,C,p)

    n,dR = pynof.ocupacion(gamma,p.no1,p.ndoc,p.nalpha,p.nv,p.nbf5,p.ndns,p.ncwo,p.HighSpin,p.occ_method)

    return E,C,gamma,n,grad,res.nit,res.success
