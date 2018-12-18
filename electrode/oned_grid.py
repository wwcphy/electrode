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

    def electrical_potential(self, x, typ='dc', deri=0):
        # x: array_like, shape (1,)s

        # out = np.zeros((x.shape[0], 2*derivative+1), np.double)
        x = self.get_x(x)[None,:]
        dat = getattr(self, typ+'_data', None)[deri]
        weight = getattr(self, typ, 1.)
        out = weight*map_coordinates(dat, x.T, order=1, mode="nearest")
        return out

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

        dc = self.electrical_potential(x, "dc", deri)
        rf = self.pseudo_potential(x, deri)
        if typ == 'tot':
            return dc + rf
        elif typ == 'dc':
            return dc
        elif typ == 'rf':
            return rf
        else:
            raise ValueError("Valid typ: 'tot','dc','rf'.")

    def secularf(self, qoverm, l):
        omega2 = qoverm*(2*(self.rf_coeff*l)**2*self.rf**2-self.dc)
        if omega2 > 0:
            return np.sqrt(omega2)
        else:
            print("Can't be trapped in pseudo-potential.")

    def trajectory_toy(self, x0, v0, qoverm, omega, t0=np.double(0.),
        dt=np.double(1e-5), t1=1e4, nsteps=1, phi=0., itern=3, *args, **kwargs):
        t, ndt = np.double(t0), np.double(dt)*nsteps
        # t, x0, v0 = np.double(t0), np.array(x0,np.double)[axis], np.array(v0,np.double)[axis]
        yi = np.array([x0,v0])
        def solve_toy(t_span, y0, *args, **kwargs):
            dt = t_span[1]-t_span[0]
            xpi, vpi = y0[0],y0[1]
            if kwargs.get("pseudo",False):
                api = -qoverm*self.potential(xpi, deri=1)[0]
                xp, vp, ap = xpi, vpi, api
                for i in range(itern):
                    dx, dv = vp*dt, ap*dt    # +1/2*ap*dt**2
                    xp, vp = (2*xpi+dx)/2, (2*vpi+dv)/2
                    ap = (2*api-qoverm*self.potential(xp, deri=1)[0])/2
            else:
                api = -qoverm*self.time_potential(xpi, deri=1, t=omega*t, phi=phi)[0]
                xp, vp, ap = xpi, vpi, api
                for i in range(itern):
                    dx, dv = vp*dt, ap*dt    # +1/2*ap*dt**2
                    # print(dx, dv)
                    xp, vp = (2*xpi+dx)/2, (2*vpi+dv)/2
                    ap = (2*api-qoverm*self.time_potential(xpi, deri=1, t=omega*(t+ndt), phi=phi)[0])/2
            return np.array([dx,dv])
        while t < t1:
            sol_dy = solve_toy(t_span=(t, t+ndt), y0=yi, *args, **kwargs)
            t += ndt
            yi += sol_dy
            yield t, yi[0], yi[1]    # aviod changing yi, the ouput is along axis



