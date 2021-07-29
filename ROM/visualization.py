""" @author: Saeed  """

import os, os.path

import matplotlib as matplt
import matplotlib.pyplot as plt
from matplotlib import animation

import numpy as np

def contourPlot(X, Y, phi, fileName='filename', figSize=(14,7)):

    fig, axs = plt.subplots(1,1,figsize=figSize)
    barRange = np.linspace(0.0, 1.0, 30, endpoint=True)

    axs.contour(X, Y, phi, barRange, linewidths=0.5, colors='k')
    cntr1 = axs.contourf(X, Y, phi, barRange, cmap="RdBu_r")
    cb = fig.colorbar(cntr1, ax=axs, shrink=0.8, orientation='vertical')
    axs.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(fileName, bbox_inches = 'tight', pad_inches = 0.1, dpi = 400)

    fig.clear(True)

def contourSubPlot(X, Y, phi1, phi2, phi3, phi4, fileName='filename', figSize=(14,7)):

    fig, axs = plt.subplots(2,2,figsize=figSize)
    #fig.suptitle(plotTitle)
    #barRange = np.linspace(0.0, 1.0, 30, endpoint=True)
    barRange = np.linspace(-0.60, 0.5, 30, endpoint=True)
    phi = [phi1, phi2, phi3, phi4]
    i = 0
    for ax in axs.flat:
        ax.contour(X, Y, phi[i], barRange, linewidths=0.5, colors='k')
        cntr1 = ax.contourf(X, Y, phi[i], barRange, cmap="RdBu_r")
        ax.set_aspect('equal')
        i = i + 1
    cb1 = fig.colorbar(cntr1, ax=axs, shrink=0.8, orientation='vertical')

    #fig.tight_layout()
    fig.savefig(fileName, bbox_inches = 'tight', pad_inches = 0.1, dpi = 400)

    fig.clear(True)

def animationGif(X, Y, phi, fileName='filename', figSize=(14,7)):

    fig = plt.figure(figsize=figSize)

    plt.xticks([])
    plt.yticks([])
    
    def animate(i):
        cont = plt.contourf(X,Y,phi[:,:,i],120,cmap='jet')
        return cont  
    
    anim = animation.FuncAnimation(fig, animate, frames=50)
    fig.tight_layout()
    # anim.save('animation.mp4')
    writergif = animation.PillowWriter(fps=10)
    anim.save(fileName+'.gif',writer=writergif)

    fig.clear(True)

def plot1(figNum, epochs, loss, valLoss, label1, label2, label3, label4, plotTitle, fileName):

    fig = plt.figure(figNum)
    plt.plot(epochs, loss)
    plt.plot(epochs, valLoss)
    plt.xlabel(label3)
    plt.ylabel(label4)
    plt.legend([label1, label2])
    plt.title(plotTitle)
    plt.savefig(fileName)
    fig.clear(True)

def plot(figNum, epochs, loss, valLoss, label1, label2, plotTitle, fileName):

    fig = plt.figure(figNum)
    plt.semilogy(epochs, loss, 'b', label=label1)
    plt.semilogy(epochs, valLoss, 'r', label=label2)
    plt.title(plotTitle)
    plt.legend()
    plt.savefig(fileName)
    fig.clear(True)

def subplot(figNum, epochs, predData, trueData, label1, label2, label3, label4, plotTitle, fileName):

    fig, axs = plt.subplots(3, 3)
    fig.suptitle(plotTitle)

    i = 0
    for ax in axs.flat:
        ax.plot(epochs, trueData[:, i])
        ax.plot(epochs, predData[:, i])
        if i%3 == 0:
            ax.set_ylabel(label4)
        if i%3 == 1:
            ax.set_xlabel(label3)
        ax.legend([label1, label2], loc=1, prop={'size': 6})
        i = i + 1

    plt.savefig(fileName)
    fig.clear(True)




    

