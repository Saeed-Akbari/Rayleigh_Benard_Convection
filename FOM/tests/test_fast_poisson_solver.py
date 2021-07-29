#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 14:14:41 2021

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from poisson_solvers.fast_poisson_solvers import *

#%%
if __name__ == "__main__":
    with open(r'poisson_solver.yaml') as file:
        input_data = yaml.load(file, Loader=yaml.FullLoader)    
    file.close()


    for i in range(3):
        nx = 16*(2**i)
        ny = 16*(2**i)
    
        
        x_l = input_data['x_l']
        x_r = input_data['x_r']
        y_b = input_data['y_b']
        y_t = input_data['y_t']
        ipr = input_data['ipr']
        ips = input_data['ips']

        dx = (x_r-x_l)/nx
        dy = (y_t-y_b)/ny
        
        x = np.linspace(x_l, x_r, nx+1)
        y = np.linspace(y_b, y_t, ny+1)
        
        xm, ym = np.meshgrid(x,y, indexing='ij')
        
        if ipr == 1:
            ue = np.sin(2.0*np.pi*xm)*np.sin(2.0*np.pi*ym)    
            f = -8.0*np.pi**2*np.sin(2.0*np.pi*xm)*np.sin(2.0*np.pi*ym)
        elif ipr == 2:
            km = 16.0
            c1 = (1.0/km)**2
            c2 = -2.0*np.pi**2
            
            ue = np.sin(2.0*np.pi*xm)*np.sin(2.0*np.pi*ym) + \
                  c1*np.sin(km*np.pi*xm)*np.sin(km*np.pi*ym)
            
            f = 4.0*c2*np.sin(2.0*np.pi*xm)*np.sin(2.0*np.pi*ym) + \
                c2*np.sin(km*np.pi*xm)*np.sin(km*np.pi*ym)
        
        if ips == 1:
            un = spectral(nx,ny,dx,dy,f)
        elif ips == 2:
            un = fst(nx,ny,dx,dy,f)
        elif ips == 3:
            un = fst4(nx,ny,dx,dy,f)
        elif ips == 4:
            un = fst4_tdma(nx,ny,dx,dy,f)
        elif ips == 5:
            un = fps(nx,ny,dx,dy,f)
        elif ips == 6:
            un = fps4_tdma(nx,ny,dx,dy,f)

        errL2 = np.linalg.norm(un - ue)/np.sqrt(np.size(ue))
        
        print('#----------------------------------------#')
        print('n = %d' % (nx))
        print('L2 error:  %5.3e' % errL2)
        if i>=1:
            rateL2 = np.log(errL2_0/errL2)/np.log(2.0);
            print('L2 order:  %5.3f' % rateL2)
        
        errL2_0 = errL2
        
    fig, axs = plt.subplots(1,2,figsize=(14,5))
    cs = axs[0].contourf(xm, ym, ue, 60,cmap='jet')
    #cax = fig.add_axes([1.05, 0.25, 0.05, 0.5])
    fig.colorbar(cs, ax=axs[0], orientation='vertical')
    
    cs = axs[1].contourf(xm, ym, un,60,cmap='jet')
    #cax = fig.add_axes([1.05, 0.25, 0.05, 0.5])
    fig.colorbar(cs, ax=axs[1], orientation='vertical')
    
    plt.show()
    fig.tight_layout()
        