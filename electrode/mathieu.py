# wwc 12/14/2018
# Mathieu special function class.

import warnings

import numpy as np
from scipy.integrate import solve_ivp

class Mathieu():
    """Generate bounded Mathieu solutions, with non-integer
    characteristic exponent nu dependent on given a, q.

    """
    def nu_exp_det(self,a,q,ntr):
        """characteristic exponent nu using Cosisson2009
        truncated determinant method. Most precise.

        Parameters
        ----------
        ntr : int > 0
            Truncated dimension of delta0 determinant.

        """
        delta0 = np.zeros((2*ntr+1,2*ntr+1))
        for n in range(ntr):
            delta0[n+ntr,n+ntr] = 1.
            delta0[n+ntr,n+ntr+1], delta0[n+ntr,n+ntr-1] = q/(4*n**2-a), q/(4*n**2-a)
            delta0[ntr-n,ntr-n] = 1.
            delta0[ntr-n,ntr-n+1], delta0[ntr-n,ntr-n-1] = q/(4*n**2-a), q/(4*n**2-a)
        delta0[0,0], delta0[ntr+ntr,ntr+ntr] = 1., 1.
        delta0[0,1], delta0[ntr+ntr,ntr+ntr-1] = q/(4*ntr**2-a), q/(4*ntr**2-a)
        d0 = np.linalg.det(delta0)
        # print(d0)
        rhs = d0*np.sin(np.pi/2*np.sqrt(a))**2
        if rhs < 0 or rhs > 1:
            nu = 2./np.pi*np.arcsinh(np.sqrt(np.complex(-rhs)))    # In fact, this is mu.
            warnings.warn('Complex characteristic exponent nu: unbounded motion.')
        else:
            nu = 2./np.pi*np.arcsin(np.sqrt(rhs))   # mu = j*nu
        return nu

    def nu_exp_d0ap(self,a,q):
        """characteristic exponent nu using EqWorld(ode0234)
        and Abramowitz and Stegun (1964) Eq.(20.3.18) method.
        Fast and precise.

        """
        cosnu = np.cos(np.pi*np.sqrt(a)) \
            + np.pi*q**2/(4*np.sqrt(a)*(a-1))*np.sin(np.pi*np.sqrt(a)) \
            + q**4*(np.pi**4/96.-25*np.pi**2/256.)
        return np.arccos(cosnu)/np.pi

    def nu_exp_bla(self,a,q):
        """characteristic exponent nu approximation when
        |a| < q**2 << 1. See blakested2010. Not precise.

        """
        return np.sqrt(a+q**2/2)

    def nu_exp_ivp(self,a,q,nstep=None,method='RK45'):
        """characteristic exponent nu using NIST https://dlmf.nist.gov/28.34#i
        and Abramowitz and Stegun (1964) Eq.(20.3.10) method.
        Much slower but precise.

        Parameters
        ----------
        nstep : int > 0
            Steps for t_eval argument of solve_ivp
        method : sting
            'method' argument of solve_ivp
            'RK45' is faster but 'Radau' is more precise. 

        """
        def func(t,y):
            # y[0,1] = [y,dy/dt]
            dy0 = y[1]
            dy1 = -(a-2*q*np.cos(2*t))*y[0]
            return [dy0,dy1]
        if nstep != None:
            t_eval = np.linspace(0.,np.pi,nstep+1)
        else:
            t_eval = None
        sol = solve_ivp(func,[0.,np.pi],[1.,0.],method=method,t_eval=t_eval)
        if np.abs(sol.y[0,-1]) > 1.:
            nu = np.arccosh(np.complex(sol.y[0,-1]))/np.pi    # In fact, this is mu.
            warnings.warn('Complex characteristic exponent nu: unbounded motion.')
        else:
            nu = np.arccos(sol.y[0,-1])/np.pi   # mu = j*nu
        return nu

    def nu_tolerance(self,a,q,tol=1e-5,nstart=5):
        """Error estimate of characteristic exponent nu.

        Parameters
        ----------
        tol : float
            |nu_n-nu_2n| < tol
        nstart : int > 0
            Dimension to start nu_exp_det()

        Returns
        -------
        float, int
            nu and order that satisfy tolerance.
        """
        cycle = 100
        for i in range(cycle):
            n = nstart*(i+1)
            nu_n = self.nu_exp_det(a,q,n)
            nu_2n = self.nu_exp_det(a,q,n*2)
            if type(nu_n) == complex:
                warnings.warn('Complex characteristic exponent nu: unbounded motion.')
                return nu_2n, 2*n
            elif np.abs(nu_n-nu_2n) < tol:
                return nu_2n, 2*n
        raise ValueError("Can't reach %.2e precision using %d"%(tol,nstart*cycle*2)
            +" order or lower truncated determinant.")

    def gn(self,n,a,q,nu):
        """Recurrence relation of coefficients.

        """
        return (a-(nu+2*n)**2)/q

    # # Test method.
    # def coeff_ratio0(self,nc,ng,a,q,nu):
    #     # only 0 ~ nc, all use continued fraction.
    #     ratio = []
    #     for n in range(nc):
    #         # continuous fraction starts from n+1, cuts off at ng+n
    #         frac = 1/self.gn(ng,a,q,nu)
    #         for i in range(ng-1+n,n,-1):    # range(...) ~ [ng+n-1,...,n+1]
    #             frac = 1/(self.gn(i,a,q,nu)-frac)
    #         ratio.append(frac)
    #     return np.array(ratio)

    # # Test method.
    # def coeff_ratio1(self,nc,ng,a,q,nu):
    #     # only 0 ~ nc, use recurrence relation.
    #     frac = 1/self.gn(ng+nc,a,q,nu)
    #     for i in range(ng+nc-1,nc-1,-1):    # range(...) ~ [ng+n-1,...,n+1]
    #         frac = 1/(self.gn(i,a,q,nu)-frac)
    #     ratio = [frac]
    #     for n in range(nc-1,0,-1):
    #         # continuous fraction starts from n+1, cuts off at ng+n
    #         frac = 1/(self.gn(n,a,q,nu)-ratio[0])
    #         ratio.insert(0,frac)
    #     return np.array(ratio)

    # # Test method.
    # def coeff_C1(self,nc,ng,a,q,nu):
    #     ratio_c = self.coeff_ratio0(nc,ng,a,q,nu)
    #     cn = [1.0]
    #     for n in range(nc):
    #         cn.append(cn[n]*ratio_c[n])
    #     cn = np.array(cn)
    #     scale = 1./np.sum(cn)
    #     cn = cn*scale
    #     return cn

    def coeff_ratio(self,a,q,nu,nc,ng):
        """Generate ratio of Floquet solution coefficients 
        using continued fraction.
        Compute  C[nc]               1
                ———————  =  —————————————————————
                C[nc-1]                   1
                            G[nc] - —————————————
                                    G[nc+1] - ...
        truncated at G_(nc+ng), then use recurrence relation
        C[n±1]            1
        ——————  =  ——————————————— to get other ratio.
         C[n]               C[n±2]
                   G[n±1] - ——————
                            C[n±1]

        Parameters
        ----------
        nu : float
            characteristic exponent nu
        nc : int
            Truncated order of coefficients
        ng : int
            Truncated order of gn continued fraction

        Returns
        -------
        Ratios : array, array, shape (1, nc+1)
            Negetive and positive ratio arrays.
        """
        fracp, fracn = 1/self.gn(ng+nc,a,q,nu), 1/self.gn(-ng-nc,a,q,nu)
        for i in range(-ng-nc+1,-nc+1):    # range(...) ~ [ng+n-1,...,n+1]
            fracp, fracn = 1/(self.gn(-i,a,q,nu)-fracp), 1/(self.gn(i,a,q,nu)-fracn)
        rp, rn = [fracp],[fracn]
        for n in range(-nc+1,0):
            # continuous fraction starts from n+1, cuts off at ng+n
            fracp = 1/(self.gn(-n,a,q,nu)-rp[0])
            fracn = 1/(self.gn(n,a,q,nu)-rn[-1])
            rp.insert(0,fracp)
            rn.append(fracn)
        return np.array(rn), np.array(rp)

    def coeff_Cn(self,a,q,nu,nc,ng):
        """Generate even/odd Floquet solution coefficients
        C[-nc] to C[nc] by mutiplying self.coeff_ratio. 
        ΣC[n] (even solution) and Σ(nu+2n)*C[n] (odd solution)
        are scaled to 1.

        Parameters
        ----------
        See self.coeff_ratio.

        Returns
        -------
        Ccn, Csn, cn : array, array, array. shape (1, 2*nc+1)
            Scaled even Ccn, Scaled odd Csn, original cn.
        """
        rn, rp = self.coeff_ratio(a,q,nu,nc,ng)
        cnp, cnn = [1.0], [1.0]
        for n in range(nc):
            cnp.append(cnp[-1]*rp[n])
            cnn.insert(0,cnn[0]*rn[-n-1])
        cnn.pop(-1)
        cnn.extend(cnp)
        cn = np.array(cnn)
        vn = 2*np.linspace(-nc,nc,2*nc+1)+nu
        scale_c, scale_s = 1./np.sum(cn), 1./np.sum(vn*cn)
        Ccn, Csn = cn*scale_c, cn*scale_s
        return Ccn, Csn, cn

    def sum_c(self,a,q,nu,nc,ng):
        """Summation of coefficients.

        """
        # make it a test in future.
        Ccn, Csn = self.coeff_Cn(a,q,nu,nc,ng)[0:2]
        totn, totnu, totg, totgn = 0., 0., 0., 0.
        tots, totsnu = 0., 0.
        for n in range(-nc,nc+1):
            totn += n*Ccn[n+nc]
            totnu += (nu+2*n)*Ccn[n+nc]
            totg += self.gn(n,a,q,nu)*Ccn[n+nc]
            totgn += self.gn(n,a,q,nu)*n*Ccn[n+nc]
            tots += Csn[n+nc]
            totsnu += (nu+2*n)*Csn[n+nc]
        print(totgn-2*totn)
        print((totsnu*Ccn[nc]/Csn[nc]-nu)/2-totn)
        return totn, totnu, totg, totgn, tots, totsnu

    def init_sol(self,a,q,nc=None,ng=5,tol_nu=1e-5,mode='det'):
        """Initial computation of exponent nu and coefficients.
        Call this method before calling sol_y1 and sol_y2.

        Parameters
        ----------
        nc : int
            Truncated order of coefficients
        ng : int
            Truncated order of gn continued fraction
        tol_nu : float
            Tolerance of exponent nu.
        mode : 'det', 'd0ap' or 'ivp'
            Different method to compute exponent nu.

        """
        a, q = np.double(a), np.double(q)
        if mode == 'd0ap':
            self.nu = self.nu_exp_d0ap(a,q)
        elif mode == 'det':
            self.nu, nu_n = self.nu_tolerance(a,q,tol=tol_nu)
        elif mode == 'ivp':
            self.nu = self.nu_exp_ivp(a,q,method='Radau')
        else:
            raise ValueError("Invalid characteristic exponent mode.")
        if nc == None:
            nc = nu_n
        self.Cn = self.coeff_Cn(a,q,self.nu,nc,ng)[0:2]
        return 0

    def sol_y1(self,t,deri=0):
        """Mathieu Floquet even solution.
        y1(0), y1'(0) = 1., 0.

        Parameters
        ----------
        t : float
            Time
        deri : int > 0 
            Order of derivative.

        """
        try:
            nu, Ccn = self.nu, self.Cn[0]
        except AttributeError as err:
            warnings.warn("\n\nRun <Mathieu>.init_sol() first.\n")
            raise err
        nc = Ccn.shape[0]
        t, y1t = np.double(t), 0.
        tri_fun = eval(['np.cos','np.sin','np.cos','np.sin'][deri%4])
        minus = [1,-1,-1,1][deri%4]
        for n, cn in enumerate(Ccn):
            n = n-(nc-1)/2
            y1t += cn*(nu+2*n)**deri*tri_fun((nu+2*n)*t)*minus
        return y1t

    def sol_y2(self,t,deri=0):
        """Mathieu Floquet odd solution.
        y2(0), y2'(0) = 0., 1.

        Parameters
        ----------
        t : float
            Time
        deri : int > 0 
            Order of derivative.
            
        """
        try:
            nu, Csn = self.nu, self.Cn[1]
        except AttributeError as err:
            warnings.warn("\n\nRun <Mathieu>.init_sol() first.\n")
            raise err
        nc = Csn.shape[0]
        t, y2t = np.double(t), 0.
        tri_fun = eval(['np.sin','np.cos','np.sin','np.cos'][deri%4])
        minus = [1,1,-1,-1][deri%4]
        for n, cn in enumerate(Csn):
            n = n-(nc-1)/2
            y2t += cn*(nu+2*n)**deri*tri_fun((nu+2*n)*t)*minus
        return y2t

    def init_value(self,y0=(0.,1.),t0=0.):
        """Solve superposition coefficients A, B of even/odd solution
        from initial value y(t0) and y'(t0).

        Parameters
        ----------
        y0 : array_like, shape (1, 2)
            [y(t0), y'(t0)]
        t0 : float
            Start time.

        Returns
        -------
        A, B : array, shape (1,2)
        """
        if t0 == 0.:
            return np.array(y0)
        else:
            den = self.sol_y1(t0,0)*self.sol_y2(t0,1)-self.sol_y1(t0,1)*self.sol_y2(t0,0)
            A = (y0[0]*self.sol_y2(t0,1)-y0[1]*self.sol_y2(t0,0))/den
            B = -(y0[0]*self.sol_y1(t0,1)-y0[1]*self.sol_y1(t0,0))/den
            return np.array(A,B)

    def sol_plot():
        pass