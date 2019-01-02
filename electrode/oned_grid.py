# check numerical trajectory. wwc 12/04/2018

import warnings

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.integrate import solve_ivp

def func(x):
    """
    Quadratic potential
    """
    # x = np.float_(x)
    return x*x

def to_txt(data):
    """
    """
    file = open('grid_data.txt','w')
    for pot,fie in zip(*data):
        file.write(str(pot)+'\t'+str(fie)+'\n')
    file.close()

class System_grid():
    
    def __init__(self, grid, **kwargs):
        self.grid = np.array(grid)
        self.spacing = self.grid[1]-self.grid[0]
        self.origin = 0.0-(self.grid.shape[0]-1)/2*self.spacing
        self.dc, self.rf = 1.0, 1.0

    def generate_pot(self,func,typ='dc'):
        """
        grid : array_like
            Equal spacing 1D coordinate x.
        func : callable
            Potential function
        """
        # grid = self.grid
        pot_data = func(self.grid)
        field_data = np.gradient(pot_data,self.spacing)  # leave out '-' on purpose
        pot2_data = np.gradient(field_data,self.spacing)
        setattr(self, typ+'_data', [pot_data, field_data, pot2_data])
        return [pot_data, field_data, pot2_data]
        # self.dc_data = ...

    def get_x(self,x):
        return np.array([(x-self.origin)/self.spacing])

    def rf_scale(self, qoverm, o, l):
        self.rf_coeff = np.double(np.sqrt(qoverm)/(2*o*l))
        return self.rf_coeff

    def analytic(self,x,deri=0):
        if deri == 0:
            return np.double(self.dc/2.*x*x)
        elif deri == 1:
            return np.double(self.dc*x)

    def electrical_potential(self, x, typ='dc', deri=0):
        # x: array_like, shape (1,)s

        # out = np.zeros((x.shape[0], 2*derivative+1), np.double)
        x = self.get_x(x)[None,:]
        dat = getattr(self, typ+'_data', None)[deri]
        weight = getattr(self, typ, 1.)
        out = weight*map_coordinates(dat, x.T, order=1, mode="nearest")
        return out[0]

    def time_potential(self, x, deri=0, t=0., phi=0.):

        t = np.double(t)
        dc = self.electrical_potential(x, typ='dc', deri=deri)
        rf = self.electrical_potential(x, typ='rf', deri=deri)
        return dc + rf*np.cos(t+phi)

    def pseudo_potential(self, x, deri=0):

        try:
            p = [self.rf_coeff*self.electrical_potential(x, "rf", i)
                for i in range(1, deri+2)]
        except AttributeError as err:
            warnings.warn("\n\nHaven't set rf_coeff. Run <system>.rf_scale() first.\n")
            raise err
        if deri == 0:
            return p[0]**2    # p[0] is real field
        elif deri== 1:
            return 2*p[0]*p[1]
        else:
            raise ValueError("only know how to generate pseudopotentials "
                "up to 1st order")

    def potential(self, x, deri=0, typ='tot'):

        if typ == 'tot':
            dc = self.electrical_potential(x, "dc", deri)
            rf = self.pseudo_potential(x, deri)
            return dc + rf
        elif typ == 'dc':
            dc = self.electrical_potential(x, "dc", deri)
            return dc
        elif typ == 'rf':
            rf = self.pseudo_potential(x, deri)
            return rf
        else:
            raise ValueError("Valid typ: 'tot','dc','rf'.")

    def secularf(self, qoverm, l):
        omega2 = qoverm*(2*(self.rf_coeff*l)**2*self.rf**2-self.dc)
        if omega2 > 0:
            return np.sqrt(omega2)
        else:
            print("Can't be trapped in pseudo-potential.")

    def trajectory_euler(self, x0, v0, qoverm, t0=0., dt=1e-5,
        t1=1e4, nsteps=1, itern=3, mode='analytic'):
        t, ndt = np.double(t0), np.double(dt)*nsteps
        yi = np.array([x0,v0])
        if mode == 'discrete':
            mode = 'electrical_potential'
        def solve_euler(dt, y0):
            xpi, vpi = y0
            api = -qoverm*getattr(self,mode)(xpi, deri=1)
            xp, vp, ap = xpi, vpi, api
            for i in range(itern):
                vp = vpi+(api+ap)/2.*dt
                xp = xpi+(vpi+vp)/2.*dt
                ap = -qoverm*getattr(self,mode)(xp, deri=1)
            return np.array([xp,vp])
        while t < t1:
            t, yi = t+ndt, solve_euler(dt=ndt, y0=yi)
            yield t, yi[0], yi[1]

    def trajectory_RK(self, x0, v0, qoverm, t0=0., dt=1e-5,
        t1=1e4, nsteps=1, integ="RK45", mode='analytic', *args, **kwargs):

        from scipy.integrate import solve_ivp

        t, ndt = np.double(t0), np.double(dt)*nsteps
        yi = np.array([x0,v0])
        if mode == 'discrete':
            mode = 'electrical_potential'
        # kwargs.setdefault('t_eval',np.linspace(t,t+ndt,nsteps+1))
        def ddx(t, y):
            vp, ap = y[1], -qoverm*getattr(self,mode)(y[0], deri=1)
            return np.array([vp,ap])
        while t < t1:
            # use result of last nstep as input
            sol = solve_ivp(ddx, t_span=(t, t+ndt), y0=yi, method=integ, *args, **kwargs)
            t, yi = t+ndt, sol.y[:,-1]
            # kwargs['t_eval'] += ndt
            yield t, yi[0], yi[1]

    def trajectory_toy(self, x0, v0, qoverm, omega=None, t0=0., dt=1e-5,
        t1=1e4, nsteps=1, phi=0., itern=3, pseudo=True, typ='tot', *args, **kwargs):
        t, ndt = np.double(t0), np.double(dt)*nsteps
        yi = np.array([x0,v0])
        # This iteration of solve_toy has a wrong format.
        def solve_toy(t_span, y0, *args, **kwargs):
            dt = t_span[1]-t_span[0]
            xpi, vpi = y0[0],y0[1]
            if pseudo == True:
                api = -qoverm*self.potential(xpi, deri=1, typ=typ)
                xp, vp, ap = xpi, vpi, api
                for i in range(itern):
                    dx, dv = vp*dt, ap*dt    # +1/2*ap*dt**2
                    xp, vp = xpi+dx, (2*vpi+dv)/2
                    ap = (2*api-qoverm*self.potential(xp, deri=1, typ=typ))/2
            else:
                api = -qoverm*self.time_potential(xpi, deri=1, t=omega*t, phi=phi)
                xp, vp, ap = xpi, vpi, api
                for i in range(itern):
                    dx, dv = vp*dt, ap*dt    # +1/2*ap*dt**2
                    xp, vp = (2*xpi+dx)/2, (2*vpi+dv)/2
                    ap = (2*api-qoverm*self.time_potential(xpi, deri=1, t=omega*(t+ndt), phi=phi))/2
            return np.array([dx,dv])
        while t < t1:
            sol_dy = solve_toy(t_span=(t, t+ndt), y0=yi, *args, **kwargs)
            t += ndt
            yi += sol_dy
            yield t, yi[0], yi[1]

    def trajectory(self, x0, v0, qoverm, omega=None, t0=0., dt=.0063*2*np.pi,
        t1=1e4, nsteps=1, phi=0., integ="RK45", pseudo=True, typ='tot',
        *args, **kwargs):
        """Calculate an ion trajectory."""

        from scipy.integrate import solve_ivp

        t, ndt = np.double(t0), np.double(dt)*nsteps
        yi = np.array([x0,v0])
        # kwargs.setdefault('t_eval',np.linspace(t,t+ndt,nsteps+1))
        def ddx(t, y):
            xp, vi = y[0], y[1]
            if pseudo == True:
                ai = -qoverm*self.potential(xp, deri=1,typ=typ)
            elif kwargs.get("analytic",False):
                ai = -qoverm*analytic(xp)
            else:
                ai = -qoverm*self.time_potential(xp, deri=1, t=omega*t, phi=phi)
            return np.array([vi,ai])
        while t < t1:
            # use result of last nstep as input
            sol = solve_ivp(ddx, t_span=(t, t+ndt), y0=yi, method=integ, *args, **kwargs)
            t += ndt
            # kwargs['t_eval'] += ndt
            yi = sol.y[:,-1]    # -1: y(t+ndt)
            yield t, yi[0], yi[1]