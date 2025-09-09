import numpy as np
from scipy.optimize import minimize
from time import time
import pynof 

def hfidr(C,H,I,b_mnl,E_nuc,p,printmode):

    no1_ori = p.no1
    p.no1 = p.nbeta

    n = np.zeros((p.nbf5))
    n[0:p.nbeta] = 1.0
    n[p.nbeta:p.nalpha] = 0.5

    cj12 = 2*np.einsum('i,j->ij',n,n)
    ck12 = np.einsum('i,j->ij',n,n)
    if(p.MSpin==0 and p.nsoc>1):
        ck12[p.nbeta:p.nalpha,p.nbeta:p.nalpha] = 2*np.einsum('i,j->ij',n[p.nbeta:p.nalpha],n[p.nbeta:p.nalpha])

    if(printmode):
        print("Hartree-Fock")
        print("============")
        print("")

        print('{:^7} {:^7} {:^14} {:^14} {:^15} {:^14}'.format("Nitext","Nitint","Eelec","Etot","Ediff","maxdiff"))

    E,elag,sumdiff,maxdiff = pynof.ENERGY1r(C,n,H,I,b_mnl,cj12,ck12,p)

    fmiug0 = None

    ext = True
    # iteraciones externas
    for i_ext in range(p.maxitid):
        if i_ext==0:
            maxlp = 1
        else:
            maxlp = p.maxloop

        # iteraciones internas
        for i_int in range(maxlp):
            E_old = E

            if(p.scaling):
                fmiug = pynof.fmiug_scaling(fmiug0,elag,i_ext,p.nzeros,p.nbf,p.noptorb)

            fmiug0, W = np.linalg.eigh(fmiug)
            C = np.matmul(C,W)
            E,elag,sumdiff,maxdiff = pynof.ENERGY1r(C,n,H,I,b_mnl,cj12,ck12,p)

            E_diff = E-E_old
            if(abs(E_diff)<p.thresheid):
                if(printmode):
                    print('{:6d} {:6d} {:14.8f} {:14.8f} {:14.8f} {:14.8f}'.format(i_ext,i_int,E,E+E_nuc,E_diff,maxdiff))
                for i in range(p.nbf):
                    fmiug0[i] = elag[i][i]
                ext = False
                break

        if(not ext):
            break
        if(printmode):
            print('{:6d} {:6d} {:14.8f} {:14.8f} {:14.8f} {:14.8f}'.format(i_ext,i_int,E,E+E_nuc,E_diff,maxdiff))


    # Regresamos no1 a su estado original
    p.no1 = no1_ori

    return E,C,fmiug0

def occoptr(gamma,C,H,I,b_mnl,p):

    J_MO,K_MO,H_core = pynof.computeJKH_MO(C,H,I,b_mnl,p)

    E = 0
    nit = 0
    success = True

    if (p.ndoc>0 and not p.freeze_occ):
        res = minimize(pynof.calcocce, gamma, args=(J_MO,K_MO,H_core,p), jac=pynof.calcoccg, method=p.occupation_optimizer)
        gamma = res.x
        E = res.fun
        nit = res.nit
        success = res.success

    n,dR = pynof.ocupacion(gamma,p.no1,p.ndoc,p.nalpha,p.nv,p.nbf5,p.ndns,p.ncwo,p.HighSpin,p.occ_method)
    cj12,ck12 = pynof.PNOFi_selector(n,p)

    return E,nit,success,gamma,n,cj12,ck12

def orboptr(C,n,H,I,b_mnl,cj12,ck12,i_ext,itlim,fmiug0,p,printmode):

    i_int = 0
    success_orb = False

    E,elag,sumdiff,maxdiff = pynof.ENERGY1r(C,n,H,I,b_mnl,cj12,ck12,p)

    if(maxdiff<p.threshl):# and abs(E_diff)<p.threshe):
        success_orb = True
        return E,C,i_int,success_orb,itlim,fmiug0

    if (p.scaling and i_ext>1 and i_ext >= itlim):# and sumdiff > sumdiff_old):
        p.nzeros = p.nzeros + 1
        itlim = i_ext + p.itziter
        if (p.nzeros>p.nzerosm):
            p.nzeros = p.nzerosr
        #if (p.nzeros>abs(int(np.log10(maxdiff)))+1):
        #    p.nzeros = p.nzerosr
            #p.nzeros = abs(int(np.log10(maxdiff)))
    sumdiff_old = sumdiff

    if i_ext==0:
        maxlp = 1
    else:
        maxlp = p.maxloop

    fmiug = np.zeros((p.noptorb,p.noptorb))
    fk = np.zeros((30,p.noptorb,p.noptorb))
    bdiis = np.zeros((31,31))
    cdiis = np.zeros((31))
    iloop = 0
    idiis = 0

    for i_int in range(maxlp):
        iloop = iloop + 1
        E_old2 = E
        
        #scaling
        if(p.scaling):
            fmiug = pynof.fmiug_scaling(fmiug0,elag,i_ext,p.nzeros,p.nbf,p.noptorb)
        if(p.diis and maxdiff < p.thdiis):
            fk,fmiug,idiis,bdiis = pynof.fmiug_diis(fk,fmiug,idiis,bdiis,cdiis,maxdiff,p.noptorb,p.ndiis,p.perdiis)

        eigval, eigvec = np.linalg.eigh(fmiug)
        fmiug0 = eigval

        C = np.matmul(C,eigvec)

        E,elag,sumdiff,maxdiff = pynof.ENERGY1r(C,n,H,I,b_mnl,cj12,ck12,p)

        E_diff2 = E-E_old2

        if(abs(E_diff2)<p.threshec or i_int==maxlp-1):
            break

    return E,C,i_int,success_orb,itlim,fmiug0

def orbopt_rotations(gamma,C,H,I,b_mnl,p):

    y = np.zeros((p.nvar))

    n,dn_dgamma = pynof.ocupacion(gamma,p.no1,p.ndoc,p.nalpha,p.nv,p.nbf5,p.ndns,p.ncwo,p.HighSpin,p.occ_method)
    cj12,ck12 = pynof.PNOFi_selector(n,p)

    if("trust" in p.orbital_optimizer or "Newton-CG" in p.orbital_optimizer):
        res = minimize(pynof.calcorbeg, y, args=(n,cj12,ck12,C,H,I,b_mnl,p),jac=True,hess="2-point",method=p.orbital_optimizer,options={"maxiter":p.maxloop})
    else:
        res = minimize(pynof.calcorbeg, y, args=(n,cj12,ck12,C,H,I,b_mnl,p),jac=True,method=p.orbital_optimizer,options={"maxiter":p.maxloop})

    E = res.fun
    y = res.x
    grad = res.jac
    nit = res.nit
    success = res.success

    C = pynof.rotate_orbital(y,C,p)

    return E,C,nit,success


def orbopt_adam2(gamma,C,H,I,b_mnl,p):

    n,dn_dgamma = pynof.ocupacion(gamma,p.no1,p.ndoc,p.nalpha,p.nv,p.nbf5,p.ndns,p.ncwo,p.HighSpin,p.occ_method)
    cj12,ck12 = pynof.PNOFi_selector(n,p)
    y = np.zeros((p.nvar))
    E = pynof.calcorbe(y, n,cj12,ck12,C,H,I,b_mnl,p)

    alpha = p.alpha
    beta1 = 0.7
    beta2 = 0.999

    y = np.zeros((p.nvar))

    m = 0.0 * y
    v = 0.0 * y
    vhat_max = 0.0 * y

    improved = False
    success = False
    best_E, best_C = E, C
    nit = 0

    for i in range(p.maxloop):
        nit += 1

        grad = pynof.calcorbg(y*0, n,cj12,ck12,C,H,I,b_mnl,p)

        if np.linalg.norm(grad) < 10**-4 and improved:
            success = True
            break

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad**2)
        mhat = m / (1.0 - beta1**(i+1))
        vhat = v / (1.0 - beta2**(i+1))
        vhat_max = np.maximum(vhat_max, vhat)
        y = - alpha * mhat / (np.sqrt(vhat_max + 10**-8)) #AMSgrad
        C = pynof.rotate_orbital(y,C,p)

        E = pynof.calcorbe(y*0, n,cj12,ck12,C,H,I,b_mnl,p)
        #print(i," ",E," ", E < best_E)
        if E < best_E:
            best_C = C
            best_E = E
            improved = True

    if not improved:
        p.alpha = p.alpha/10
        p.maxloop = p.maxloop + 30
        #print("      alpha ",p.alpha)

    return best_E,best_C,nit,success


def orbopt_adam(gamma,C,H,I,b_mnl,p):

    n,dn_dgamma = pynof.ocupacion(gamma,p.no1,p.ndoc,p.nalpha,p.nv,p.nbf5,p.ndns,p.ncwo,p.HighSpin,p.occ_method)
    cj12,ck12 = pynof.PNOFi_selector(n,p)
    y = np.zeros((p.nvar))
    E = pynof.calcorbe(y, n,cj12,ck12,C,H,I,b_mnl,p)

    alpha = p.alpha
    beta1 = 0.7
    beta2 = 0.999

    y = np.zeros((p.nvar))

    m = 0.0 * y
    v = 0.0 * y
    vhat_max = 0.0 * y

    improved = False
    success = False
    best_E, best_C = E, C
    nit = 0

    for i in range(p.maxloop):
        nit += 1

        grad = pynof.calcorbg(y*0, n,cj12,ck12,C,H,I,b_mnl,p)

        if np.linalg.norm(grad) < 10**-4 and improved:
            success = True
            break

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad**2)
        mhat = m / (1.0 - beta1**(i+1))
        vhat = v / (1.0 - beta2**(i+1))
        vhat_max = np.maximum(vhat_max, vhat)
        y = - alpha * mhat / (np.sqrt(vhat_max + 10**-8)) #AMSgrad
        C = pynof.rotate_orbital(y,C,p)

        E = pynof.calcorbe(y*0, n,cj12,ck12,C,H,I,b_mnl,p)
        #print(i," ",E," ", E < best_E)
        if E < best_E:
            best_C = C
            best_E = E
            improved = True

    return best_E,best_C,nit,success

def orbopt_sd(gamma,C,H,I,b_mnl,p):
     '''  IN: 
          gamma: Occupation Numbers (ONs)
          C: NOs guess
          H: Monoelectronic Integrals
          I: Repulsion Integrals
          b_mnl: RI approximation Integrals
          p: molecule properties
       OUT:
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

def orbopt_cg(gamma,C,H,I,b_mnl,p):
     '''  IN: 
          gamma: Occupation Numbers (ONs)
          C: NOs guess
          H: Monoelectronic Integrals
          I: Repulsion Integrals
          b_mnl: RI approximation Integrals
          p: molecule properties
       OUT:
    '''
     
     print("Starting SD ... with alpha",p.alpha)
     ### We calculate the occupations numbers of the Orbitals
     n,dn_dgamma = pynof.ocupacion(gamma,p.no1,p.ndoc,p.nalpha,p.nv,p.nbf5,p.ndns,p.ncwo,p.HighSpin,p.occ_method)
     ## NOF Coefficients, realted with the selected functional
     cj12,ck12 = pynof.PNOFi_selector(n,p)
     y = np.zeros((p.nvar))
     # We calculate the Energy
     E = pynof.calcorbe(y, n,cj12,ck12,C,H,I,b_mnl,p)
     ### onjugated Gradient
     step_size=p.alpha
     grad_old = None
     d = None

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
              
          if grad_old is None:
               d = -grad  # **Primera iteración: dirección de descenso = gradiente negativo**
          else:
               beta = np.dot(grad.flatten(), (grad - grad_old).flatten()) / np.dot(grad_old.flatten(), grad_old.flatten())  # **Fórmula de Polak-Ribière**
               d = -grad + beta * d  # **Actualiza dirección conjugada**

          grad_old = grad.copy()  # **Guarda el gradiente actual**

          y = step_size * d  # **Nuevo paso en dirección conjugada** 
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

def orbopt_adagrad(gamma,C,H,I,b_mnl,p):
     '''  converges in 16716   steps.    '''
     n,dn_dgamma = pynof.ocupacion(gamma,p.no1,p.ndoc,p.nalpha,p.nv,p.nbf5,p.ndns,p.ncwo,p.HighSpin,p.occ_method)
     cj12,ck12 = pynof.PNOFi_selector(n,p)
     y = np.zeros((p.nvar))
     E = pynof.calcorbe(y, n,cj12,ck12,C,H,I,b_mnl,p)

     y = np.zeros((p.nvar))
     sq_grad_sums = np.zeros((p.nvar))
     # step_size=0.1
     step_size=p.alpha

     improved = False
     success = False
     best_E, best_C = E, C
     nit = 0

     for i in range(p.maxloop):
         nit += 1
         sq_grad_sums = C.shape[0]
         
         grad = pynof.calcorbg(y*0, n,cj12,ck12,C,H,I,b_mnl,p)

         sq_grad_sums += grad**2.0
         
         if np.linalg.norm(grad) < 10**-4 and improved:
             success = True
             break
         # calculate the step size for this variable
         alpha = step_size / (1e-8 + np.sqrt(sq_grad_sums))
         y = - alpha * grad
         
         C = pynof.rotate_orbital(y,C,p)

         E = pynof.calcorbe(y*0, n,cj12,ck12,C,H,I,b_mnl,p)
         #print(i," ",E," ", E < best_E)
         if E < best_E:
             best_C = C
             best_E = E
             improved = True

     return best_E,best_C,nit,success


def orbopt_rmsprop(gamma,C,H,I,b_mnl,p):
     ''' Converges in 15 Steeps. E = -76.24171605 '''

     n,dn_dgamma = pynof.ocupacion(gamma,p.no1,p.ndoc,p.nalpha,p.nv,p.nbf5,p.ndns,p.ncwo,p.HighSpin,p.occ_method)
     cj12,ck12 = pynof.PNOFi_selector(n,p)
     y = np.zeros((p.nvar))
     E = pynof.calcorbe(y, n,cj12,ck12,C,H,I,b_mnl,p)
 

     y = np.zeros((p.nvar))
     sq_grad_avg = np.zeros((p.nvar))
     
     # Good Value: step_size=0.00025
# Hyperparameters
     step_size=p.alpha
     #decay
     rho = 0.9
     sq_grad_avg = 0     # Initialize moving average of squared gradients
     epsilon = 1e-8      # For numerical stability

     improved = False
     success = False
     best_E, best_C = E, C
     nit = 0
     
     print('@NEW@')

     for i in range(p.maxloop):
         nit += 1

         grad = pynof.calcorbg(y * 0, n, cj12, ck12, C, H, I, b_mnl, p)

         # Update moving average of squared gradients
         if isinstance(sq_grad_avg, int):
             sq_grad_avg = grad**2
         else:
             sq_grad_avg = rho * sq_grad_avg + (1 - rho) * grad**2

         # Check convergence
         if np.linalg.norm(grad) < 1e-4 and improved:
             success = True
             break

         # Compute adaptive learning rate
         alpha = step_size / (np.sqrt(sq_grad_avg) + epsilon)

         # Update step
         y = -alpha * grad
         C = pynof.rotate_orbital(y, C, p)

         # Evaluate energy
         E = pynof.calcorbe(y * 0, n, cj12, ck12, C, H, I, b_mnl, p)


         if E < best_E:
             best_C = C
             best_E = E
             improved = True

     return best_E,best_C,nit,success


def orbopt_rmsprop_old(gamma,C,H,I,b_mnl,p):
     ''' Converges in 15 Steeps. E = -76.24171605 '''

     n,dn_dgamma = pynof.ocupacion(gamma,p.no1,p.ndoc,p.nalpha,p.nv,p.nbf5,p.ndns,p.ncwo,p.HighSpin,p.occ_method)
     cj12,ck12 = pynof.PNOFi_selector(n,p)
     y = np.zeros((p.nvar))
     E = pynof.calcorbe(y, n,cj12,ck12,C,H,I,b_mnl,p)
 

     y = np.zeros((p.nvar))
     sq_grad_avg = np.zeros((p.nvar))
     
     # Good Value: step_size=0.00025
     step_size=p.alpha
     #decay
     rho = 0.999

     improved = False
     success = False
     best_E, best_C = E, C
     nit = 0

     #p.maxloop = 10
     for i in range(p.maxloop):
     #for i in range(1000):
         nit += 1
         grad = pynof.calcorbg(y*0, n,cj12,ck12,C,H,I,b_mnl,p)
         
         if np.linalg.norm(grad) < 10**-4 : #and improved:
             success = True
             break
             
         # calculate the squared gradient        
         sg = grad**2.0
         # update the moving average of the squared gradient
         sq_grad_avg = (sq_grad_avg * rho) + ((1.0-rho)*sg)
         alpha = step_size / (1e-6 + np.sqrt(sq_grad_avg))
         y = - alpha * grad
         C = pynof.rotate_orbital(y,C,p)

         E = pynof.calcorbe(y*0, n,cj12,ck12,C,H,I,b_mnl,p)
         #print(i," ",E," ", E < best_E)
         if E < best_E:
             best_C = C
             best_E = E
             improved = True

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
    '''
    
    ### We calculate the occupations numbers of the Orbitals
    n, dn_dgamma = pynof.ocupacion(gamma, p.no1, p.ndoc, p.nalpha, p.nv,p.nbf5, p.ndns, p.ncwo, p.HighSpin, p.occ_method)
    ## NOF Coefficients, realted with the selected functional
    cj12, ck12 = pynof.PNOFi_selector(n, p)
    y = np.zeros(p.nvar)
     # We calculate the Energy
    E = pynof.calcorbe(y, n, cj12, ck12, C, H, I, b_mnl, p)
    ############

    # Initailize accumulation variables and define parameters
    rho=0.90          # decay factor
    epsilon=1e-5      # avoid dividing by zero
    alpha_gl=p.alpha     # scale the step to avoid being stuck at the first one (≈ “learning rate”)
    theta_max=0.30     # maximum rotation angle (rad)
    
    Eg2  = np.zeros_like(y)
    Edx2 = np.zeros_like(y) 

    #E is the energy
    #######
    #C are the coefficients of the orbitals we want to optimize
    best_E, best_C = E, C
    #####
     
    nit = 0

    improved = False
    success = False

    
    for i in range(p.maxloop):
        # we calculate the orbital gradient
        grad = pynof.calcorbg(y*0, n,cj12,ck12,C,H,I,b_mnl,p)

        # Check convergence
        if np.linalg.norm(grad) < 10**-4 and improved:
            success = True
            break


        # Implement Adadelta
        Eg2  = rho * Eg2  + (1 - rho) * grad**2
        RMS_g = np.sqrt(Eg2  + epsilon)
        RMS_dx = np.sqrt(Edx2 + epsilon)
        y = -alpha_gl * (RMS_dx / RMS_g) * grad  # introduced learning rate 
        Edx2 = rho * Edx2 + (1 - rho) * y**2

        # Limit the angle 
        norm_y = np.linalg.norm(y)
        if norm_y > theta_max:
            y *= theta_max / norm_y

       # Get the new coefficients of the orbitals based on the step = y, the Old coefficients (C), 
        #and the properties of our system (p)
        C = pynof.rotate_orbital(y,C,p)
       # Calculate the new energy
        E = pynof.calcorbe(y*0, n,cj12,ck12,C,H,I,b_mnl,p)

        # Keeps the best result
        if E < best_E:
            best_E, best_C = E, C
    
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
