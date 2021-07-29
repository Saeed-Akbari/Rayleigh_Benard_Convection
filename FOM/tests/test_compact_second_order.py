#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 14:14:41 2021

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt 

font = {'family' : 'Times New Roman',
     'size'   : 18}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

#import rhs_schemes.compact_schemes_first_order_derivative

from rhs_schemes.compact_schemes_second_order_derivative import *

if __name__ == "__main__":
    xl = -1.0
    xr = 1.0
    
    grid = []
    error = []
    
    print('#-----------------Dxx-------------------#')
    for i in range(5):
#        dx = 0.1/(2**i)
        nx = 32*2**i #int((xr - xl)/dx)
        dx = (xr - xl)/nx
          
        ny = nx
        dy = (xr - xl)/ny
        
        x = np.linspace(xl,xr,nx+1)
        y = np.linspace(xl,xr,ny+1)
        X,Y = np.meshgrid(x,y, indexing='ij')
        xx = (np.ones((ny+1,1))*x).T
        
        u = np.zeros((nx+1,ny+1))
        uddx = np.zeros((nx+1,ny+1))
        uddn = np.zeros((nx+1,ny+1))
        
        u[:,:] = np.sin(np.pi*X) + np.sin(2.0*np.pi*Y)
        uddx[:,:] = -(np.pi**2)*np.sin(np.pi*X) 
        
        uddn = c4dd(u,dx,dy,nx,ny,'XX')
        
        errL2 = np.linalg.norm(uddx - uddn)/np.sqrt(np.size(uddn))
        
        error.append(errL2 )
        grid.append(nx )
        
        print('#----------------------------------------#')
        print('n = %d' % (nx))
        print('L2 error:  %5.3e' % errL2)
        if i>=1:
            rateL2 = np.log(errL2_0/errL2)/np.log(2.0);
            print('L2 order:  %5.3f' % rateL2)
        
        errL2_0 = errL2
    
    print('#-----------------Dyy-------------------#')
    for i in range(3):
#        dx = 0.1/(2**i)
        nx = 32*2**i #int((xr - xl)/dx)
        dx = (xr - xl)/nx
          
        ny = int(0.5*nx)
        dy = (xr - xl)/ny
        
        x = np.linspace(xl,xr,nx+1)
        y = np.linspace(xl,xr,ny+1)
        X,Y = np.meshgrid(x,y, indexing='ij')
        xx = (np.ones((ny+1,1))*x).T
        
        u = np.zeros((nx+1,ny+1))
        uddy = np.zeros((nx+1,ny+1))
        uddn = np.zeros((nx+1,ny+1))
        
        u[:,:] = np.sin(np.pi*X) + np.sin(2.0*np.pi*Y)
        uddy[:,:] = -(4.0*np.pi**2)*np.sin(2.0*np.pi*Y) 
        
        uddn = c4dd(u,dx,dy,nx,ny,'YY')
        
        errL2 = np.linalg.norm(uddy - uddn)/np.sqrt(np.size(uddn))
        
        print('#----------------------------------------#')
        print('n = %d' % (nx))
        print('L2 error:  %5.3e' % errL2)
        if i>=1:
            rateL2 = np.log(errL2_0/errL2)/np.log(2.0);
            print('L2 order:  %5.3f' % rateL2)
        
        errL2_0 = errL2
    
    #%%    
    fig, axs = plt.subplots(1,1,figsize=(6,5))
    grid = np.array(grid)
    line = 100000*grid**(-4.0)
    
    axs.loglog(grid, error, 'ro-', fillstyle='none', ms = 8 )
    axs.loglog(grid, line, 'k--' )
    
    axs.text(2**6,7e-3, '$-\epsilon^{4}$')
    axs.set_xscale('log', basex=2)
    axs.grid()    
    axs.set_title('Second-order')
    axs.set_xlabel('$N$')
    axs.set_ylabel('$\epsilon$')
    plt.show()
    fig.tight_layout()
    fig.savefig('s_order.png', dpi=200)
    
#    fig, axs = plt.subplots(1,2,figsize=(14,5))
#    cs = axs[0].contourf(X, Y, uddy, 60,cmap='jet')
#    #cax = fig.add_axes([1.05, 0.25, 0.05, 0.5])
#    fig.colorbar(cs, ax=axs[0], orientation='vertical')
#    
#    cs = axs[1].contourf(X, Y, uddn,60,cmap='jet')
#    #cax = fig.add_axes([1.05, 0.25, 0.05, 0.5])
#    fig.colorbar(cs, ax=axs[1], orientation='vertical')
#    
#    plt.show()
#    fig.tight_layout()