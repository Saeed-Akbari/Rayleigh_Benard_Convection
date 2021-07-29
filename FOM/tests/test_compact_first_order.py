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
import time

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt 

font = {'family' : 'Times New Roman',
     'size'   : 18}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

#import rhs_schemes.compact_schemes_first_order_derivative

from rhs_schemes.compact_schemes_first_order_derivative import *

#%%
if __name__ == "__main__":
    xl = -1.0
    xr = 1.0
    
    grid = []
    error = []
    
    start = time.time()
    
    print('#-----------------Dx-------------------#')
    for i in range(5):
        # dx = 0.05/(2**i)
        nx = 32*2**i #int((xr - xl)/dx)
        dx = (xr - xl)/nx
          
        ny = 2*nx
        dy = (xr - xl)/ny
        
        x = np.linspace(xl,xr,nx+1)
        y = np.linspace(xl,xr,ny+1)
        X,Y = np.meshgrid(x,y, indexing='ij')
        xx = (np.ones((ny+1,1))*x).T
        
        u = np.zeros((nx+1,ny+1))
        udx = np.zeros((nx+1,ny+1))
        udn = np.zeros((nx+1,ny+1))
        
        u[:,:] = np.sin(np.pi*X) + np.sin(2.0*np.pi*Y)
        udx[:,:] = (np.pi)*np.cos(np.pi*X) 
        
        udn = c4d(u,dx,dy,nx,ny,'X')
#        udn = c4d_b4(u,dx,nx,ny,'X')
        
        errL2 = np.linalg.norm(udx - udn)/np.sqrt(np.size(udn))
        
        error.append(errL2 )
        grid.append(nx )
        
        print('#----------------------------------------#')
        print('n = %d' % (nx))
        print('L2 error:  %5.3e' % errL2)
        if i>=1:
            rateL2 = np.log((errL2_0)/(errL2))/np.log(2.0);
            print('L2 order:  %5.3f' % rateL2)
        
        errL2_0 = errL2
    
    print('#-----------------Dy-------------------#')
    for i in range(4):
#        dx = 0.05/(2**i)
        nx = 32*2**i #int((xr - xl)/dx)
        dx = (xr - xl)/nx
          
        ny = int(1*nx)
        dy = (xr - xl)/ny
        
        x = np.linspace(xl,xr,nx+1)
        y = np.linspace(xl,xr,ny+1)
        X,Y = np.meshgrid(x,y, indexing='ij')
        xx = (np.ones((ny+1,1))*x).T
        
        u = np.zeros((nx+1,ny+1))
        udy = np.zeros((nx+1,ny+1))
        udn = np.zeros((nx+1,ny+1))
        
        u[:,:] = np.sin(np.pi*X) + np.sin(2.0*np.pi*Y)
        udy[:,:] = (2.0*np.pi)*np.cos(2.0*np.pi*Y) 
        
        udn = c4d(u,dx,dy,nx,ny,'Y')
#        udn = c4d_b4(u,dx,ny,nx,'Y')
                
        errL2 = np.linalg.norm(udy - udn)/np.sqrt(np.size(udn))
        
        print('#----------------------------------------#')
        print('n = %d' % (nx))
        print('L2 error:  %5.3e' % errL2)
        if i>=1:
            rateL2 = np.log(errL2_0/errL2)/np.log(2.0);
            print('L2 order:  %5.3f' % rateL2)
        
        errL2_0 = errL2
    
    print('CPU time = ', time.time() - start)

#%%    
    fig, axs = plt.subplots(1,1,figsize=(6,5))
    grid = np.array(grid)
    line = 500*grid**(-4.0)
    
    axs.loglog(grid, error, 'ro-', fillstyle='none', ms = 8 )
    axs.loglog(grid, line, 'k--' )
    
    axs.text(2**6,5e-5, '$-\epsilon^{4}$')
    axs.set_xscale('log', basex=2)
    axs.grid()    
    axs.set_title('First-order')
    axs.set_xlabel('$N$')
    axs.set_ylabel('$\epsilon$')
    plt.show()
    fig.tight_layout()
    fig.savefig('f_order.png', dpi=200)
    
#    fig, axs = plt.subplots(1,2,figsize=(14,5))
#    cs = axs[0].contourf(X, Y, udy, 60,cmap='jet')
#    #cax = fig.add_axes([1.05, 0.25, 0.05, 0.5])
#    fig.colorbar(cs, ax=axs[0], orientation='vertical')
#    
#    cs = axs[1].contourf(X, Y, udn,60,cmap='jet')
#    #cax = fig.add_axes([1.05, 0.25, 0.05, 0.5])
#    fig.colorbar(cs, ax=axs[1], orientation='vertical')
#    
#    plt.show()
#    fig.tight_layout()