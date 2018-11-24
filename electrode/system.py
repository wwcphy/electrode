# -*- coding: utf8 -*-
#
#   electrode: numeric tools for Paul traps
#
#   Copyright (C) 2011-2012 Robert Jordens <jordens@phys.ethz.ch>
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import (absolute_import, print_function,
        unicode_literals, division)

import warnings, itertools
from contextlib import contextmanager
import logging

import numpy as np
from scipy import optimize, constants as ct
import matplotlib.pyplot as plt

if not hasattr(optimize, "minimize"):  # Check if scipy.optimize has attribute "minimize"
    # quick work around for scipy<0.11  # Perhaps this is an alternative method for old version scipy. wwc
    class _Result(object):
        pass
    def minimize(fun, x0, jac=None, hess=None, *args, **kwargs):
        method = kwargs.pop("method", "Newton-CG")
        assert method == "Newton-CG"
        r = optimize.fmin_ncg(f=fun, x0=x0, fprime=jac,    # see scipy.optimize.fmin_ncg()  wwc
                fhess=hess, full_output=True, *args, **kwargs)
        res = _Result()
        res.x, res.success, res.message = r[0], r[5] == 0, "unknown"
        return res
    optimize.minimize = minimize    # change name.  wwc

try:
    import cvxopt, cvxopt.modeling
except ImportError:
    warnings.warn("cvxopt not found, optimizations will fail", ImportWarning)
    cvxopt = None

try:
    from qc.theory.gni import gni
except ImportError:
    warnings.warn("qc modules not found, some stuff will fail", ImportWarning)
    gni = None

from .transformations import euler_from_matrix
from .saddle import rfo
from .electrode import PolygonPixelElectrode
from .utils import (expand_tensor, norm, rotate_tensor,
    mathieu, name_to_deriv)
from .pattern_constraints import (PatternRangeConstraint,
        PotentialObjective)
from . import colors


logger = logging.getLogger("electrode")


class System(list):
    """A collection of Electrodes.

    Parameters
    ----------
    electrodes : list of `Electrode`
        Individual Electrodes comprising this System.
    """
    def __init__(self, electrodes=[], **kwargs):
        super(System, self).__init__(**kwargs)
        self.extend(electrodes)
    
    @property
    def names(self):
        """List of names of the electrodes.
        
        This property can be set but in-place changes have no effect.
        """
        return [el.name for el in self]
   
    @names.setter
    def names(self, names):
        for ei, ni in zip(self, names):
            ei.name = ni
    
    @property
    def dcs(self):
        """Array of `dc` potentials of the electrodes.
        
        This property can be set but in-place changes have no effect.
        """
        return np.array([el.dc for el in self])

    @dcs.setter
    def dcs(self, voltages):
        for ei, vi in zip(self, voltages):
            ei.dc = vi

    @property
    def rfs(self):
        """Array of `rf` potentials of the electrodes. 
        
        This property can be set but in-place changes have no effect.
        """
        return np.array([el.rf for el in self])

    @rfs.setter
    def rfs(self, voltages):
        for ei, vi in zip(self, voltages):
            ei.rf = vi

    def __getitem__(self, name_or_index):
        """Electrode lookup.
        
        Returns
        -------
        Electrode
            The electrode given by its index or name.
            None if not found by name.
            # E.g. for instance s, s[0] or s["DC1"] both refer to the "GridElectrode" instance of DC1. wwc

        Raises
        ------
        IndexError
            If electrode index does not exist.
        """
        try:
            return list.__getitem__(self, name_or_index)
        except TypeError:
            for ei in self:
                if ei.name == name_or_index:
                    return ei

    electrode = __getitem__

    @contextmanager
    def with_voltages(self, dcs=None, rfs=None):
        """Returns a `contextmanager` with temporary voltage setting.
        
        This is a convenient way to temporarily change the voltages and
        ensure they are reset to their old values.

        Parameters
        ----------
        dcs : array_like
            `dc` voltages for all electrodes, or None to keep present
            ones.
        rfs : array_like
            `rf` voltages for all electrodes, or None to keep present
            ones.
            
        Returns
        -------
        contextmanager
        
        Examples
        --------
        >>> s = System()
        >>> with s.with_voltages(dcs=0*s.dcs, rfs=[0, 1]):
        ...     print(s.potential([0, 0, 1.]))
        """
        try:
            if dcs is not None:
                dcs, self.dcs = self.dcs, dcs
            if rfs is not None:
                rfs, self.rfs = self.rfs, rfs
            yield
        finally:
            if dcs is not None:
                self.dcs = dcs
            if rfs is not None:
                self.rfs = rfs

    def electrical_potential(self, x, typ="dc", derivative=0, expand=False):
        """Electrical potential derivative.
        # Not only for "dc", but also used in pseudo_potential when typ="rf"  wwc
        # Because it calls GridElectrode.potential(), the pot return from this method is scaled with amplitude.  wwc

        Parameters
        ----------
        x : array_like, shape (n, 3)
            Positions to evaluate the potential at.
        typ : {"dc", "rf"}
            Potential to scale the electrodes contribution with.
        derivative : int
            Derivative order.
        expand : bool
            If True, return the fully expanded tensor, else return the
            reduced form.

        Returns
        -------
        potential : array
            Potential at `x`.
            If `expand == False`, shape (n, l) and `l` is the derivative
            index, `l = 2*derivative + 1`. Else, shape (n, 3, ..., 3) and
            returns the fully expanded tensorial form.

        See Also
        --------
        utils.expand_tensor
        utils.select_tensor
            Utility functions to convert between the reduced and
            expanded tensorial forms.
        """
        x = np.asanyarray(x, dtype=np.double).reshape(-1, 3)
        pot = np.zeros((x.shape[0], 2*derivative+1), np.double)
        for ei in self:  # self (System class, inherited from list) instance contains multiple GridElectrode (ei) instance.  wwc
            # getattr check if typ is an attribute of ei (GridElectrode instance), if it is, return the value of ei.typ  wwc
            # Typically, ei.typ is ei.dc or ei.rf, which are the scaling factor of each electrode potential.  wwc
            vi = getattr(ei, typ, None)
            if vi:  # If set a ei.vi (ei.dc/ei.rf) = 0, you won't go into this if.  wwc
                # ei can be any "electrode" classes in electrode.py, but we only use GridElectrode, # wwc
                # so ei.potential refers to line ~595 in electrode.py.  wwc
                ei.potential(x, derivative, potential=vi, out=pot)    # "potential" is the argument for ei.potential to scale.  wwc  Tried 100.
                # print(ei.name,"value:", vi)    # wwc
                # All dc potentials have been summed in the last for loop of ei.potential--out--pot  wwc
        if expand:    # "expand" decides if we will reshape potential array. In self.pseudo and self.potential, all True.  wwc
            pot = expand_tensor(pot)
        return pot    # pot now is summation of all dc.  wwc
    
    def individual_potential(self, x, derivative=0):
        """Individual contributions to the electrical potential.
        
        Returns an array of the contributions by the electrodes to the
        potential at each point.  Each electrode is taken to have unit
        potential (in turn grounding all others).

        Parameters
        ----------
        x : array_like, shape (n, 3)
            Points to evaluate at.
        derivative : int
            Derivative order.

        Returns
        -------
        potentials : array, shape (m, n, l)
            Potential contributions. `m` is the electrode index (index
            into `self`). `n` is the point index, `l = 2 * derivative +
            1` is the derivative index.
        """
        x = np.asanyarray(x, dtype=np.double).reshape(-1, 3)
        eff = np.zeros((len(self), x.shape[0], 2*derivative+1),
                np.double)
        for i, ei in enumerate(self):
            ei.potential(x, derivative, potential=1., out=eff[i])
        return eff

    def time_potential(self, x, derivative=0, t=0., expand=False):
        """Electrical potential at an instant.
        
        No pseudopotential averaging. The phase of the rf voltage is
        assumed to be equal across all rf carrying electrodes and the rf
        voltage is assumed to be maximal at `t = 0`::

            e_dc + cos(t) * e_rf

        Parameters
        ----------
        derivative : int
            Derivative order
        t : float
            Time instant
        expand : bool
            Expand to full tensorial form if True

        Returns
        -------
        array
            See `electrical_potential`.
        """
        dc, rf = (self.electrical_potential(x, typ, derivative, expand)
                for typ in ("dc", "rf"))
        return dc + np.cos(t)*rf

    def pseudo_potential(self, x, derivative=0):
        """The ponderomotive/pseudo potential.
        
        Parameters
        ----------
        x : array, shape (n, 3)
            Points to evaluate the pseudopotential at
        derivative : int <= 4
            Derivative order. Implemented up to 4th order.
            
        Returns
        -------
        potential : array, shape (n, 3, ..., 3)
            Pseudopotential derivative. Fully expanded since this is not
            generally harmonic.
        """
        # RObert calculates derivates of pseudopotential here. Pseudopotential itself is
        # calculated in "if derivative==0:", the dot product of field, see blakestad2010, Eq.(5.2).
        # Other orders are derivatives for pseudopotential.  wwc
        p = [self.electrical_potential(x, "rf", i, expand=True)
                for i in range(1, derivative+2)]
        if derivative == 0:    # pseudopotential itself.
            return np.einsum("ij,ij->i", p[0], p[0])  # Einstein summation  wwc
        elif derivative == 1:    # "field" of pseudopotential. Two parts.  wwc
            return 2*np.einsum("ij,ijk->ik", p[0], p[1])
        elif derivative == 2:    # Two parts.  wwc
            return 2*(np.einsum("ijk,ijl->ikl", p[1], p[1])  # a2^2  wwc
                     +np.einsum("ij,ijkl->ikl", p[0], p[2]))  # a1*a3  wwc
        elif derivative == 3:
            a = np.einsum("ij,ijklm->iklm", p[0], p[3])
            b = np.einsum("ijk,ijlm->iklm", p[1], p[2])
            a += b
            a += b.transpose(0, 2, 1, 3)
            a += b.transpose(0, 3, 2, 1)
            return 2*a
        elif derivative == 4:
            a = np.einsum("ij,ijklmn->iklmn", p[0], p[4])
            b = np.einsum("ijk,ijlmn->iklmn", p[1], p[3])
            a += b
            a += b.transpose(0, 4, 2, 3, 1)
            a += b.transpose(0, 3, 2, 1, 4)
            a += b.transpose(0, 2, 1, 3, 4)
            c = np.einsum("ijkl,ijmn->iklmn", p[2], p[2])
            a += c
            a += c.transpose(0, 1, 4, 3, 2)
            a += c.transpose(0, 1, 3, 2, 4)
            return 2*a
        else:
            raise ValueError("only know how to generate pseudopotentials "
                "up to 4th order")

    def potential(self, x, derivative=0):
        """Combined electrical and ponderomotive potential.
        
        Parameters
        ----------
        x : array, shape (n, 3)
            Points to evaluate at.
        derivative : int <= 4
            Derivative order. Implemented up to 4th order.

        Returns
        -------
        potential : array, shape (n, 3, ..., 3)
            Pseudopotential derivative. Fully expanded since this is not
            generally harmonic.
        """
        dc = self.electrical_potential(x, "dc", derivative,
                expand=True)
        rf = self.pseudo_potential(x, derivative)
        return dc + rf    # summation of rf with all dc.  wwc

    def plot(self, ax, alpha=.3, **kwargs):
        """Plot electrodes projected onto the xy plane.
        
        Plots each electrode according to its interpretation of `plot()`
        with sequential colors into the given axes.
        
        Parameters
        ----------
        ax : matplotlib axes
        **kwargs : any
            Passed to all `Electrode.plot()`.
        """
        for e, c in zip(self, itertools.cycle(colors.set3)):
            e.plot(ax, color=tuple(c/255.), alpha=alpha, **kwargs)

    def plot_contour(self, ax, grid=None, slc='x', slc_at=0., fill=False,
            divide_max=8, line_num=50, **kwargs):
        """Contour plot total static and pseudo potential to either 
        cross-section of x, y, z. Use max and min of potential as 
        contour line regions ("levels" argument of plt.contour())

        Parameters
        ----------
        ax : matplotlib axes
        grid : Grid instance (see grid.py)
            The center, step, shape parameters of grid has to be
            the same as the grid when it's created for the calculation 
            in bem. Values are store in vtk files, find a GridElectrode
            instance e of either electrode. See below default grid.
        slc : 'x', 'y', 'z'
            The cross-section plane you want to slice.
        slc_at : float
            Slice cross-section at slc_at
        fill : bool
            Fill intervals (contourf) if True.
        divide_max : float >= 1
            Determine the upper limit of contour line region.
        line_num : int
            The number of contour lines to plot.
        **kwargs : any
            Passed all to plt.contour(), usually plot arguments.

        Returns
        -------
        maxp, minp : max and min of potential in the plot region
        """

        # Create a Grid instance by default, using grid parameters from
        # vtk of 1st GridElectrode (usually 'DC1') in system (self[0]). 
        if grid == None:
            from . import Grid
            step, shape = self[0].spacing, self[0].data[0].shape[:-1]
            # center is the midpoint of grid, so it's not self[0].origin.
            center = self[0].origin + (np.array(shape)-1)*step/2
            grid = Grid(center=center, step=step, shape=shape)

        e3 = slc_at
        sec = {'x':[1,2],'y':[0,2],'z':[0,1]}
        coord = {'x':[2,0,1],'y':[0,2,1],'z':[0,1,2]}
        xyz = grid.to_xyz()[sec[slc]]    # point arrays of the other two axes
        pot = []
        for e2 in xyz[1]:
            pot.append([self.potential(x=np.array([e1,e2,e3])[coord[slc]],derivative=0)[0]
                for e1 in xyz[0] ])
        pot = np.array(pot)
        maxp, minp = np.amax(pot),np.amin(pot)
        print("max, min potential: %f, %f"%(maxp, minp))

        axlb = {'x':'yz','y':'xz','z':'xy'}
        maxcl, mincl = maxp/divide_max, minp
        if maxcl <= mincl:
            maxcl = (maxp-minp)/2 + minp
            print("Have taken another levels upper limit. Use a smaller divide_max.")
        kwargs.setdefault('levels',np.linspace(mincl, maxcl, line_num))
        # kwargs.setdefault('cmap',plt.cm.Blues)
        # vmin = maxcl/2 can have a better colormap contrast.
        kwargs.setdefault('vmin',maxcl/2)
        ax.set_xlabel(axlb[slc][0]+'/l',fontsize=15)
        ax.set_ylabel(axlb[slc][1]+'/l',fontsize=15)
        if fill == True:
            fplot = ax.contourf
        else:
            fplot = ax.contour
        CS = fplot(xyz[0], xyz[1], pot, **kwargs)    # CS for potential colorbar
        return CS, (maxp, minp)

    def plot_voltages(self, ax, u=None, um=None, cmap=None,
            **kwargs):
        """Plot electrodes with color proportional to voltage.

        Red for positive, blue for negative.
        
        Parameters
        ----------
        ax : matplotlib axes
        u : array or None
            Voltages to use for plotting. If None, use `self.dcs`.
        um : float or None
            Maximum voltage to scale colors to. If None, use
            `max(abs(u))`.
        cmap : matplotlib color map
            Color map to use.
        **kwargs : any
            Passed to each `Electrode.plot()`.
        """
        if cmap is None:
            import matplotlib.pyplot as plt
            cmap = plt.cm.RdBu_r
        if u is None:
            u = self.dcs
        if um is None:
            um = np.fabs(u).max() or 1.
        u = (u / um + 1)/2
        #colors = np.clip((u+.5, .5-np.fabs(u), -u+.5), 0, 1).T
        colors = [cmap(ui) for ui in u]
        for el, ci in zip(self, colors):
            el.plot(ax, color=ci, **kwargs)

    def minimum(self, x0, axis=(0, 1, 2), coord=np.identity(3),
        method="Newton-CG", **kwargs):
        """Find a potential minimum.
        # Within System.potential(), Robert has combined dc and rf. But I don't find scaling factor?  wwc
        
        Parameters
        ----------
        x0 : array_like, shape (3,)
            Start point for the downhill search.
        axis : tuple of int
            Only vary the given axis in the given coordinate system.
        coord : array, shape (3, 3)
            Coordinate system to vary the axes in.
        method : str
            Method for minimization. See `scipy.opimize.minimize()` for
            possible values.
        **kwargs : any
            Passed to the minimization method.
            
        See Also
        --------
        scipy.optimize.minimize
        """
        x = np.array(x0,dtype=np.double)    # add dtype=np.double to avoid precision loss bug.  wwc
        axis = list(axis)
        def f(xi):    # potential
            x[axis] = xi
            return self.potential(np.dot(coord, x), 0)[0]  # THe second 0 is derivative.  wwc
        def g(xi):    # gradient
            x[axis] = xi
            return rotate_tensor(self.potential(np.dot(coord, x), 1),
                    coord)[0, axis]
        def h(xi):    # hessian
            x[axis] = xi
            return rotate_tensor(self.potential(np.dot(coord, x), 2),
                    coord)[0, axis][:, axis]
        #xs = optimize.fmin_bfgs(p, np.array(x0)[axis], fprime=g,
        #        disp=False)
        x0 = np.array(x0)[axis]
        res = optimize.minimize(fun=f, x0=x0, jac=g, hess=h,
            method=method, **kwargs)    # Error messages printed out should all from minimize() itself.  wwc
        if not res.success:
            raise ValueError("failed, %i, %s, %s" % (res.success,
                res.message, res))
        x[axis] = res.x
        return x

    def saddle(self, x0, axis=(0, 1, 2), coord=np.identity(3), **kw):
        """Find a saddle point using rational function optimization.

        A saddle point is a point with vanishing first derivatives and
        only one negative second normal derivative.
        
        Parameters
        ----------
        x0 : array_like, shape (3,)
            Start point for the saddlepoint search.
        axis : tuple of int
            Only vary the given axis in the given coordinate system.
        coord : array, shape (3, 3)
            Coordinate system to vary the axes in.
        **kw : any
            Passed to `rfo()`.

        See Also
        --------
        saddle.rfo
            Find Saddle points using rational function optimization.
        """
        kwargs = dict(dx_max=.1, xtol=1e-5, ftol=1e-5)
        kwargs.update(kw)
        x = np.array(x0)
        axis = list(axis)
        def f(xi):
            x[axis] = xi
            return self.potential(np.dot(coord, x), 0)[0]
        def g(xi):
            x[axis] = xi
            return rotate_tensor(self.potential(np.dot(coord, x), 1),
                    coord.T)[0, axis]
        h = rotate_tensor(self.potential(np.dot(coord, x), 2),
                coord.T)[0, axis][:, axis]
        # rational function optimization
        xs, p, ret = rfo(f, g, np.array(x0)[axis], h=h, **kwargs)
        if not ret in ("ftol", "xtol"):
            raise ValueError("%s", ((x0, axis, x, xs, p, ret),))
        # f(xs) # update x
        return x, p    # x: saddle point, p: saddle potential  wwc

    def modes(self, x, sorted=True):
        """Curvatures and eigenmode vectors.

        physical units of the trap frequenzies (Hz):
        scale = (q*u/(omega*scale))**2/(4*m)
        (scale*ew/scale**2/m)**.5/(2*pi)

        Parameters
        ----------
        x : array_like, shape (3,)
            Point to determine modes at.
        sorted : bool
            If True, return data sorted by eigenvalue in ascending
            order.

        Returns
        -------
        ew : array, shape (3,)
            Eigenmodes
        ev : array, shape (3, 3)
            Normal modes. The mode index is the second axis.
        """
        ew, ev = np.linalg.eigh(self.potential(x, 2)[0])
        if sorted:
            i = ew.argsort()
            ew, ev = ew[i], ev[:, i]
        return ew, ev

    def trajectory(self, x0, v0, qoverm, axis=(0, 1, 2),
            t0=np.float64(0.), dt=np.float64(.0063*2*np.pi), t1=1e4, nsteps=1,
            integ="RK45", *args, **kwargs):
        """Calculate an ion trajectory.
        
        Integrates the ion trajectory without the
        adiabatic/pseudopotential approximation using a symplectic
        integration scheme.
        
        Parameters
        ----------
        x0 : array_like, shape (3,)
            Initial position.
        v0 : array_like, shape (3,)
            Initial speed.
        qoverm : float
            Ratio q/m
        axis : tuple of int
            Axes to vary during the integration. If `x0` and `v0` lie in
            a symmetry plane, the perpendicular axis can be dropped.
        t0 : float
            Initial time.
        dt : float
            Time step.
        t1 : float
            Final time.
        nsteps : int
            Interval to report position and speed at. Every `nstep` time
            stap (each `dt`) is reported.

        Returns
        -------
        generator
            Yields `(t, x, v)` time, position ans speed data.
        """
        # if not callable(integ):
        #     integ_ = getattr(gni, integ)
        #     methc = kwargs.pop("methc", 2)
        #     def integ(ddx, nsteps, t, p, q, t1, *args, **kwargs):
        #         return integ_(ddx, nsteps, t, p, q, t1, methc,
        #                 *args, **kwargs)
        # axis = list(axis)
        # t, p, q = t0, v0[axis], x0[axis]
        # x0 = np.array(x0)
        # def ddx(t, q, f):
        #     x0[axis] = q
        #     f[:] = self.time_potential(x0, 1, t, expand=True)[0, axis]
        # while t < t1:
        #     integ(ddx, nsteps, t, p, q, t+dt, *args, **kwargs)
        #     t += dt
        #     yield t, q.copy(), p.copy()

    # wwc version
        from scipy.integrate import solve_ivp

        axis, dim = list(axis), len(axis)    # dim = 3, haven't implemented axis of dim < 3.
        # axis.extend([ax+dim for ax in axis])    # can't construct such axis for x+v because len(axis) could <3.
        # axisv = [ax+dim for ax in axis]
        ndt = np.float64(dt)*nsteps
        t, x0, v0 = np.float64(t0), np.array(x0,dtype=np.float64)[axis], np.array(v0,dtype=np.float64)[axis]
        yi = np.concatenate((x0,v0))
        # print(yi)
        kwargs.setdefault('t_eval',np.linspace(t,t+ndt,nsteps+1))
        def ddx(t, y):
            # y = np.array([x1,x2,x3,v1,v2,v3])[axis]
            xp, vi = y[0:dim].copy(), y[dim:2*dim].copy()    # avoid changing y
            xp[axis] = xp    # This looks like OK with dim<3, but time_potential() needs a 3D xp.
            if "pseudo" in kwargs:
                ai = -qoverm*self.potential(xp, derivative=1)[0, axis]
            else:
                ai = -qoverm*self.time_potential(xp, 1, t, expand=True)[0, axis]
            # print("ai",ai)
            return np.concatenate((vi,ai))
        while t < t1:
            # use result of last nstep as input
            # print("t_eval",kwargs["t_eval"])
            sol = solve_ivp(ddx, t_span=(t, t+ndt), y0=yi, method=integ, *args, **kwargs)
            # print("t",sol.t,"y",sol.y)
            t += ndt
            kwargs['t_eval'] += ndt
            yi = sol.y[:,-1]    # -1: y(t+ndt)
            yield t, yi[0:dim].copy(), yi[dim:2*dim].copy()    # aviod changing yi, the ouput is along axis

    # wwc toy version
    def trajectory_toy(self, x0, v0, qoverm, axis=(0, 1, 2),
            t0=np.float64(0.), dt=np.float64(1e-5), t1=1e4, nsteps=1,
            integ="RK45", *args, **kwargs):

        axis, dim = list(axis), len(axis)    # dim = 3, haven't implemented axis of dim < 3.
        ndt = np.float64(dt)*nsteps
        t, x0, v0 = np.float64(t0), np.array(x0,dtype=np.float64)[axis], np.array(v0,dtype=np.float64)[axis]
        yi = np.concatenate((x0,v0))
        def solve_toy(t_span, y0, *args, **kwargs):
            dt = t_span[1]-t_span[0]
            xp = y0[0:dim]
            ai = -qoverm*self.time_potential(xp, 1, t_span[0], expand=True)[0, axis]
            dx, dv = y0[dim:2*dim]*dt+1/2*ai*dt**2, ai*dt
            return np.concatenate((dx,dv))
        while t < t1:
            sol_dy = solve_toy(t_span=(t, t+ndt), y0=yi, *args, **kwargs)
            t += ndt
            yi += sol_dy
            yield t, yi[0:dim].copy(), yi[dim:2*dim].copy()    # aviod changing yi, the ouput is along axis
        
    def shims(self, x_coord_deriv, objectives=[], constraints=None,
            **kwargs):
        """Determine shim vectors.

        Solves the shim equations (orthogonalizes) simultaneously at
        all points for all given derivatives. 

        Parameters
        ----------
        x_coord_deriv : list of tuples (x, coord, derivative)
            `x` being array_like, shape (3,), `coord` either None or a
            array_like shape (3, 3) coordinate system rotation matrix,
            and `derivative` a string for a partial derivative.
            For possible values see `utils._derivative_names`.
        objectives : list of `pattern_constraints.Constraint`
            Additional objectives. Use this for e.g.
            `pattern_constraints.MultiPotentialObjective`.
        constraints : None or list of `pattern_constraints.Constraint`
            List of constraints. If None, the pattern electrode
            potential values are constrained between -1 and 1.
        **kwargs : any
            Passed to `self.optimize`.

        Returns
        -------
        vectors : array, shape (n, m)
            The ith row is a shim vector that achieves a unit of the ith
            constraint's effect. `n` being the number of objectives in
            the (`len(x_coord_deriv) + len(objectives)`) and m is the
            number of electrodes (`len(self)`).
        """
        obj = [PotentialObjective(x=x, derivative=deriv, value=0,
            rotation=coord) for x, coord, deriv in x_coord_deriv]
        obj += objectives
        if constraints is None:
            constraints = [PatternRangeConstraint(min=-1, max=1)]
        vectors = np.empty((len(obj), len(self)),
                np.double)
        for i, objective in enumerate(obj):
            objective.value = 1
            p, c = self.optimize(constraints+obj, verbose=False,
                    **kwargs)
            objective.value = 0
            vectors[i] = p/c
        return vectors

    def _run_cvxopt(self, obj, ctrs, verbose=True, **kwargs):
        solver = cvxopt.modeling.op(obj, ctrs)
        cvxopt.solvers.options.update(**kwargs)
        cvxopt.solvers.options["show_progress"] = verbose
        if verbose:
            logger.info("variables: %i", sum(v._size
                    for v in solver.variables()))
            logger.info("inequalities: %i", sum(v.multiplier._size
                    for v in solver.inequalities()))
            logger.info("equalities: %i", sum(v.multiplier._size
                    for v in solver.equalities()))
        solver.solve("sparse")
        if not solver.status == "optimal":
            raise ValueError("solve failed: %s" % solver.status)
        return solver

    def solve(self, local_constraints, global_constraints, verbose=True,
            **kwargs):
        """
        Optimize dc voltages at positions x to satisfy constraints.

        O(len(constraints)*len(x)*len(electrodes)) if sparse (most of the time)
    
        Parameters
        ----------
        local_constraints : list of lists of constraints
            List of constraint sets. Each set applies to a single voltage
            configuration.
        global_constraints : list of constraints
            Global constraints that apply to the entire voltage matrix.
        verbose : bool
            Passed to the solver.

        Returns
        -------
        u : array (N, M)
            Electrode potentials, N sets of local constraints, M == len(self)
        c
            Objective value
        """
        variables = [cvxopt.modeling.variable(len(self))
            for i in range(len(local_constraints))]

        obj = 0.
        ctrs = []
        for ci, vi in zip(local_constraints, variables):
            for cj in ci:
                obj = sum(coef*val for coef, val in cj.objective(self, vi))
                ctrs.extend(cj.constraints(self, vi))
        for ci in global_constraints:
            obj += sum(coef*val for coef, val in ci.objective(self, variables))
            ctrs.extend(ci.constraints(self, variables))

        solver = self._run_cvxopt(obj, ctrs, verbose, **kwargs)

        c = solver.objective.value()
        u = np.array([np.array(v.value).ravel() for v in variables])
        return u, c

    def optimize(self, constraints, rcond=1e-9, verbose=True, **kwargs):
        """Find electrode potentials that maximize given
        constraints/objectives.
        
        Parameters
        ----------
        constraints : list of `pattern_constraints.Constraint`
            Constraints and objectives.
        rcond : float
            Cutoff for small singular values. See `scipy.linalg.pinv`
            for details.
        verbose : bool
            Passed to the solver.
        
        Returns
        -------
        potentials : array, shape (n,)
            Electrode potentials that maximize the objective and
            fulfill the constraints. `n = len(self)`.
        c : float
            Solution strength. `c` times the objective value could
            be achieved using `potentials`.
        """
        p = cvxopt.modeling.variable(len(self))
        obj = []
        ctrs = []
        for ci in constraints:
            obj.extend(ci.objective(self, p))
            ctrs.extend(ci.constraints(self, p))
        B = np.array([i[0] for i in obj])
        b = np.array([i[1] for i in obj])
        # the inhomogeneous solution
        Bp = np.linalg.pinv(B, rcond=rcond)
        g = np.dot(Bp, b)
        g2 = np.inner(g, g)
        B1 = B - np.outer(b, g)/g2 # B*g_perp
        obj = cvxopt.modeling.dot(cvxopt.matrix(g), p) # maximize this
        #FIXME: there is one singular value, drop one line
        B1 = B1[:-1]
        # B*g_perp*p == 0
        ctrs.append(cvxopt.modeling.dot(cvxopt.matrix(B1.T), p) == 0.)

        solver = self._run_cvxopt(-obj, ctrs, verbose, **kwargs)

        p = np.array(p.value, np.double).ravel()
        c = np.inner(p, g)/g2
        return p, c

    def group(self, thresholds=[0], voltages=None):
        """Group electrodes by their potentials.

        Regroups all electrodes and combines those that fall in the same
        potential bin to a single electrode.

        Parameters
        ----------
        thresholds : array_like, shape (n,), float
            Bin edges. Will be flanked by `-inf, thresholds, inf`.
        voltages : None or array_like, shape (m,)
            Electrode potentials to use for binning. If None, take
            `self.dcs`.

        Returns
        -------
        System
            The new `System` instance containing the electrode groups.
            The System returned will contain `len(thresholds) + 1`
            electrodes. The ith electrode contains all electrodes `j`
            with `thresholds[i-1] <= voltages[j] < thresholds[i]`.
        """
        if voltages is None:
            voltages = self.dcs
        if thresholds is None:
            threshold = sorted(np.unique(voltages))
        ts = [-np.inf] + list(thresholds) + [np.inf]
        eles = []
        for i, (ta, tb) in enumerate(zip(ts[:-1], ts[1:])):
            good = (ta <= voltages) & (voltages < tb)
            #if not np.any(good):
            #    continue
            paths = []
            dcs = []
            rfs = []
            for j in np.argwhere(good)[:, 0]:
                el = self[j]
                paths.extend(el.paths)
                dcs.append(el.dc)
                rfs.append(el.rf)
            eles.append(PolygonPixelElectrode(paths=paths,
                dc=np.mean(dcs), rf=np.mean(rfs)))
        return System(eles)

    def mathieu(self, x, scale, r=2, sorted=True):
        """Return characteristic exponents (mode frequencies) and
        fourier components.
        
        Parameters
        ----------
        x : array_like, shape (3,)
            Position.
        scale : float
            Scale factor q/((l*o)**2*m).
        r : int
            Band cutoff.
        sorted : bool
            If True, sort modes by ascending eigenvalue.
        
        Returns
        -------
        mu : array, shape (3*(2*r + 1),)
            Eigenvalues, sorted by value if `sorted == True`.
        b : aray, shape (3*(2*r + 1), 3*(2*r + 1))
            Eigenmodes. The ith row corresponds to the ith eigenvalue.
            If each row is reshaped as `(2*r + 1, 3)`, the first axis is
            the rf frequency multiple (from `-r` to `r`) and the second
            is the spatial direction.
        """
        a = 16*scale**2*self.electrical_potential(x, "dc", 2, expand=True)[0]
        q = 8*scale*self.electrical_potential(x, "rf", 2, expand=True)[0]
        mu, b = mathieu(r, a, q)
        if sorted:
            i = mu.imag >= 0
            mu, b = mu[i], b[:, i]
            i = mu.imag.argsort()
            mu, b = mu[i], b[:, i]
        return mu/2, b

    def analyze_static(self, x, axis=(0, 1, 2),
            m=ct.atomic_mass, q=ct.elementary_charge,
            l=100e-6, o=2*np.pi*1e6, ions=1, log=None):
        """Perform an textual analysis of the potential at and around
        a point.

        Use `atomic_mass` and `elementary_charge` from `scipy.constants`
        to set `m` and `q` correctly. All constants `m, q, l, o` are
        given in SI units.

        The rf voltages, `self.rfs`, need to be set to rescaled
        voltages, while the dc voltages `self.dcs` are SI volts::

            alpha_rf = np.sqrt(q/m)/(2*l*o)
            s.rfs = u_rf_in_volt*alpha_rf
            s.dcs = u_dc_in_volt

        Parameters
        ----------
        x : array_like, shape (3,)
            Point to analyze around.
        axis : tuple of int
            Axes to vary during minimum and saddle point searches.
        m : float
            Ion mass (kg).
        q : float
            Ion charge (C)
        l : float
            Length scale (m).
        o : float
            Rf frequency (rad/s).
        ions : int
            Also analyze the modes of multiple ions.
        log : logging level
            Log the output with the given level instead of yielding
            it line by line.

        Returns
        -------
        generator
            Yields strings that can be printed or written to a file.
        """
        # I guess Robert wraps analyse_static in case that to add new functions in future. wwc
        it = self._analyze_static(x, axis, m, q, l, o, ions) 
        if log is None:
            return it
        else:
            for line in it:
                logger.log(log, line)

    def rf_scale(self, m, q, l, o):
        return np.sqrt(q/m)/(2*l*o)    # This is the square root of pesudo potential coefficient (except amplitude Vrf).  wwc

    def _analyze_static(self, x, axis=(0, 1, 2),  # x is minimum
                        m=ct.atomic_mass, q=ct.elementary_charge,
                        l=100e-6, o=2*np.pi*1e6, ions=1):
        # rf pseudopotential voltage scale
        rf_scale = self.rf_scale(m, q, l, o)
        yield "parameters:"
        yield (" f=%.3g MHz, m=%.3g amu, q=%.3g qe,"
               " l=%.3g µm, scale=%.3g V'/V_SI"
               ) % (o/(2e6*np.pi), m/ct.atomic_mass,
                    q/ct.elementary_charge, l/1e-6, rf_scale)
        yield "coordinates:"
        yield " analyze point: %s" % (x,)    # x is 
        yield "               (%s µm)" % (x*l/1e-6,)
        # Based on the minimum found in advance in jupyter, there's another minimum.  wwc
        trap = x#self.minimum(x)
        yield " minimum is at offset: %s" % (trap - x,)
        yield "                      (%s µm)" % ((trap - x)*l/1e-6,)
        # x is minimum point (just one point), so electrical_potential(x, "dc", 0) is like [[value]].  wwc
        p_dc = self.electrical_potential(x, "dc", 0)[0]
        p_rf = self.pseudo_potential(x, 0)[0]
        yield "potential:"
        yield " dc electrical: %.3g eV" % p_dc    # All dc potential at minimum x0.  wwc  %.2g previously
        yield " rf pseudo: %.3g eV" % p_rf
        try:
            xs, xsp = self.saddle(x + 0.5*x[2])    # set a small initail shift (1e-2*x[2]) to search a saddle, usually it's on x axis.  wwc
            yield " saddle offset: %s" % (xs - x,)
            yield "               (%s µm)" % ((xs - x)*l/1e-6,)
            yield " saddle height: %.3g eV" % (xsp - (p_dc + p_rf))    # (p_dc + p_rf) is the potential at x (minimum) point.  wwc
        except:
            yield " saddle not found"
        yield "force:"
        f_dc = self.electrical_potential(x, "dc", 1)[0]
        f_rf = self.pseudo_potential(x, 1)[0]
        f_tot = f_dc + f_rf     # If it's minimum, should be 0. wwc
        yield " dc electrical: %s eV/l" % (f_dc,)
        yield "               (%s eV/m)" % (f_dc/l,)
        yield " rf pseudo: %s eV/l" % (f_rf,)
        yield "           (%s eV/m)" % (f_rf/l,)
        yield " total: %s eV/l" % (f_tot,)      # wwc
        yield "           (%s eV/m)" % (f_tot/l,)    # wwc
        yield "modes:"
        curves, modes_pp = self.modes(x)    # curves are the eigenvalues of potential Hessian matrix.  wwc
        freqs_pp = np.sqrt(q*curves/m)/(2*np.pi*l)    # secular frequencies.  wwc
        mu, b = self.mathieu(x, scale=rf_scale, r=4, sorted=True)
        freqs = mu[:3].imag*o/(2*np.pi)
        modes = b[len(b)//2 - 3:len(b)//2, :3].real
        yield " pp+dc normal curvatures: %s" % curves
        yield " motion is bounded: %s" % np.allclose(mu.real, 0)
        m4 = np.eye(4, 4)
        for nj, fj, mj in (
                ("pseudopotential", freqs_pp, modes_pp),
                ("mathieu", freqs, modes)):
            yield " %s modes:" % nj
            for ci, fi, mi in zip("abc", fj, mj.T):
                yield "  %s: %.4g MHz, %s" % (ci, fi/1e6, mi)
            m4[:3, :3] = mj
            yield "  euler angles (rzxz): %s deg" % (
                    np.rad2deg(np.array(euler_from_matrix(m4, "rzxz"))))
        un = 1e-9
        se = un*self.individual_potential(x, 1)[:, 0, :]/l
        yield " heating for %.2g nV²/Hz white uncorrelated on each electrode:" % (
                un/1e-9)
        yield "  field-noise psd: %s V²/(m² Hz)" % (se**2).sum(0)
        sem = (np.dot(se, modes)**2).sum(0)
        for ci, fi, semi in zip("abc", freqs, sem):
            yield "  %s: ndot=%.4g /s, S_E*f=%.4g (V² Hz)/(m² Hz)" % (
                ci, semi*q**2/(4*m*ct.h*fi), semi*fi)
        if ions > 1:
            xi = x+np.random.randn(ions)[:, None]*1e-3
            qi = np.ones(ions)*q/(l*4*np.pi*ct.epsilon_0)**.5
            xis, cis, mis = self.ions(xi, qi)
            freqs_ppi = (cis/l**2/m)**.5/(1e6*np.pi)
            r2 = norm(xis[1]-xis[0])
            r2a = ((q*l)**2/(2*np.pi*ct.epsilon_0*q*curves[0]))**(1/3.)
            yield "two ion modes:"
            yield " separation: %.3g (%.3g µm, %.3g µm harmonic)" % (
                r2, r2*l/1e-6, r2a/1e-6)
            for ci, fi, mi in zip("abcdef", freqs_ppi, mis.transpose(2, 0, 1)):
                yield " %s: %.4g MHz, %s/%s" % (ci, fi/1e6, mi)

    def ions(self, x0, q):
        """Find minimum energy positions of an multiple ions and
        calculate normal modes.
       
        .. note:: unused, untested, probably broken

        Parameters
        ----------
        x0 : array_like, shape (n, 3)
            Initial positions of `n` ions
        q : array_like, shape (n,)
            Normalized charge to mass ratios of the ions.

        Returns
        -------
        x : array, shape (n, 3)
            Equilibrium positions.
        ew : array, shape (3*n,)
            Eigenvalues.
        ev : array, shape (3*n, 3*n)
            Normal mode vectors.
        """
        n = len(x0)
        qs = q[:, None]*q[None, :]

        def f(x0):
            x0 = x0.reshape(-1, 3)
            p0 = self.potential(x0, 0)
            x, y, z = (x0[None, :] - x0[:, None]).transpose(2, 0, 1)
            pi = .5*qs/np.ma.array(
                    x**2+y**2+z**2)**(1/2.)
            return (p0+pi.sum(-1)).sum()

        def g(x0):
            x0 = x0.reshape(-1, 3)
            p0 = self.potential(x0, 1).T
            x, y, z = (x0[None, :] - x0[:, None]).transpose(2, 0, 1)
            pi = qs*[x, y, z]/np.ma.array(
                    x**2+y**2+z**2)**(3/2.)
            return (p0+pi.sum(-1)).T.ravel()

        def h(x0):
            x0 = x0.reshape(-1, 3)
            p0 = self.potential(x0, 2).T
            x, y, z = (x0[None, :] - x0[:, None]).transpose(2, 0, 1)
            p = expand_tensor(
                (-qs*[2*x**2-y**2-z**2, 3*x*y, 3*x*z,
                    2*y**2-x**2-z**2, 3*y*z]/np.ma.array(
                    x**2+y**2+z**2)**(5/2.)).T)
            p = p.transpose(2, 0, 3, 1)
            for i, (p0i, pii) in enumerate(
                    zip(p0.transpose(2, 0, 1), p.sum(2))):
                p[i, :, i, :] += p0i-pii
            return p.reshape(p.shape[0]*p.shape[1], -1)

        with np.errstate(divide="ignore", invalid="ignore"):
            x = optimize.fmin_ncg(f=f, fprime=g, fhess=h, x0=x0.ravel(),
                disp=0)
            #print warn
            #x1, e0, e1, e2, itf, itg, warn = optimize.fmin_bfgs(
            #    f=f, fprime=g, x0=x0.ravel(), full_output=1, disp=1)
            #print (np.sort(x1)-np.sort(x))/np.sort(x)
            #x2, e0, itf, itg, warn = optimize.fmin_cg(
            #    f=f, fprime=g, x0=x0.ravel(), full_output=1, disp=1)
            #print (np.sort(x2)-np.sort(x))/np.sort(x)
            c = h(x)
        ew, ev = np.linalg.eigh(c)
        i = np.argsort(ew)
        ew, ev = ew[i], ev[i, :].reshape(n, 3, -1)
        return x.reshape(-1, 3), ew, ev
