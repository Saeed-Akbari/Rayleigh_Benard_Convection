""" @author: Saeed  """

import os, os.path

import matplotlib as matplt
import matplotlib.pyplot as plt
from matplotlib import animation

import numpy as np

def contourPlot(X, Y, phi, barRange, fileName='filename', figSize=(14,7)):

    fig, axs = plt.subplots(1,1,figsize=figSize)

    axs.contour(X, Y, phi, barRange, linewidths=0.5, colors='k')
    cntr1 = axs.contourf(X, Y, phi, barRange, cmap="RdBu_r")
    cb = fig.colorbar(cntr1, ax=axs, shrink=0.8, orientation='vertical')
    axs.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(fileName, bbox_inches = 'tight', pad_inches = 0.1, dpi = 400)

    fig.clear(True)

def contourSubPlot(X, Y, phi1, phi2, phi3, phi4, barRange, fileName='filename', figSize=(14,7)):

    fig, axs = plt.subplots(2,2,figsize=figSize)
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

def plotPODcontent(RICd, AEmode, dirPlot):

    fig = plt.figure()
        
    nrplot = 2 * np.min(np.argwhere(RICd>99.9))
    index = np.arange(1,nrplot+1)
    newRICd = [0, *RICd]
    plt.plot(range(len(newRICd)), newRICd, 'k')
    x1 = 1
    x2 = AEmode
    y1 = RICd[x2-1]
    y2 = RICd[x2-1]
    plt.plot([x1,x2], [y1,y2], color='k', linestyle='--')
    xave = np.exp(0.75*np.log(x1) + 0.25*np.log(x2))
    plt.text(xave, y2+0.4, r'$'+str(np.round(RICd[x2-1],decimals=2))+'\%$',fontsize=10)
    plt.fill_between(index[:x2], RICd[:x2], RICd[0],alpha=0.6,color='orange')

    x1 = AEmode
    x2 = AEmode
    y1 = RICd[0]
    y2 = RICd[x2-1]
    plt.plot([x1,x2], [y1,y2], color='k', linestyle='--')
    yave = np.exp(0.5*np.log(y1) + 0.5*np.log(y2))
    plt.text(x2+0.1, yave, r'r=$'+str(x2)+'$',fontsize=10)

    x1 = 1
    x2 = np.min(np.argwhere(RICd>99.9))
    y1 = RICd[x2-1]
    y2 = RICd[x2-1]
    plt.plot([x1,x2], [y1,y2], color='k', linestyle='--')
    xave = np.exp(0.5*np.log(x1) + 0.5*np.log(x2))
    plt.text(xave, y2+0.4, r'$'+str(np.round(RICd[x2-1],decimals=2))+'\%$',fontsize=10)
    plt.fill_between(index[AEmode-1:x2], RICd[AEmode-1:x2], RICd[0],alpha=0.2,color='blue')

    x1 = np.min(np.argwhere(RICd>99.9))
    x2 = np.min(np.argwhere(RICd>99.9))
    y1 = RICd[0]
    y2 = RICd[x2-1]
    plt.plot([x1,x2], [y1,y2], color='k', linestyle='--')
    yave = np.exp(0.5*np.log(y1) + 0.5*np.log(y2))
    plt.text(x2+2, yave, r'r=$'+str(x2)+'$',fontsize=10)
    
    plt.xscale("log")
    plt.xlabel(r'\bf POD index ($k$)')
    plt.ylabel(r'\bf RIC ($\%$)')
    plt.gca().set_ylim([RICd[0], RICd[-1]+3])
    plt.gca().set_xlim([1, x2*2])
    plt.savefig(dirPlot + '/content', dpi = 500, bbox_inches = 'tight')
    #print("99.9: ", np.min(np.argwhere(RICd>99.9)))
    print("two modes: ", RICd[1])
    fig.clear(True)

    fig, ax = plt.subplots()
    ax.clear()

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
    
    fig, ax = plt.subplots()
    ax.clear()

def plot(figNum, epochs, loss, valLoss, label1, label2, plotTitle, fileName):

    fig = plt.figure(figNum)
    plt.semilogy(epochs, loss, 'b', label=label1)
    plt.semilogy(epochs, valLoss, 'r', label=label2)
    plt.title(plotTitle)
    plt.legend()
    plt.savefig(fileName)
    fig.clear(True)

    fig, ax = plt.subplots()
    ax.clear()

def subplot(figNum, epochs, predData, trueData, label1, label2, label3, label4, plotTitle, fileName, px, py):

    fig, axs = plt.subplots(px, py, figsize=(18, 9))
    fig.suptitle(plotTitle)

    i = 0
    for ax in axs.flat:
        ax.plot(epochs, trueData[:, i])
        ax.plot(epochs, predData[:, i])
        #if i%px == 0:
        #    ax.set_ylabel(label4)
        #if i%py == 1:
        #    ax.set_xlabel(label3)
        if i == 0:
            ax.legend([label1, label2], loc=0, prop={'size': 6})
        i = i + 1

    plt.savefig(fileName)
    fig.clear(True)
'''
def subplotProbe(figNum, epochs, trueData, predData, label1, label2, label3, label4, plotTitle, fileName, mesh, lx, ly):

    px, py = len(lx), len(ly)
    fig, axs = plt.subplots(px, py, figsize=(14, 7))
    fig.suptitle(plotTitle)

    i, j = 0, 0
    for ax in axs.flat:
        ax.plot(epochs, predData[:, i, j])
        ax.plot(epochs, trueData[:, i, j])
        ax.set_title(f'({(lx[i]/8)*2:.2f}, {(ly[j]/4)*1:.2f})', fontsize=10)
        #if i%px == 0:
        #    ax.set_ylabel(label4)
        #if i%py == 1:
        #    ax.set_xlabel(label3)
        if (i == 0) and (j == 0):
            ax.legend([label1, label2], loc=0, prop={'size': 6})
        j = j + 1
        if j == py:
            j = 0
            i = i + 1

    fig.tight_layout()
    plt.savefig(fileName)
    fig.clear(True)
'''

'''
def subplotProbe(epochs, data, label1, label2, plotTitle, fileName, lx, ly):

    px, py = len(lx), len(ly)
    fig, axs = plt.subplots(px, py, figsize=(14, 7))
    fig.suptitle(plotTitle)

    i, j = 0, 0
    for ax in axs.flat:
        ax.plot(epochs, data[:, i, j])
        ax.set_title(f'({(lx[i]/8)*2:.2f}, {(ly[j]/4)*1:.2f})', fontsize=10)
        if (i == 0) and (j == 0):
            ax.legend([label1, label2], loc=0, prop={'size': 6})
        j = j + 1
        if j == py:
            j = 0
            i = i + 1

    fig.tight_layout()
    plt.savefig(fileName)
    fig.clear(True)    
'''


#from matplotlib.ticker import FormatStrFormatter
#import matplotlib as mpl
#mpl.rc('text', usetex=True)
#mpl.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
#mpl.rcParams['text.latex.preamble'] = r'\boldmath'
#font = {'family' : 'normal',
#        'weight' : 'bold',
#        'size'   : 20}
#mpl.rc('font', **font)

def subplotMode(figNum, epochsTest, epochsTrain, trueData, testData, trainData,\
                trueLabel, testLabel, trainLabel, fileName, px, py):

    fig, axs = plt.subplots(px, py, figsize=(18,8))

    i = 0
    for ax in axs.flat:
        ax.plot(epochsTrain,trainData[:, i], label=r'\bf{{{}}}'.format(trainLabel), linewidth = 3)
        ax.plot(epochsTest,testData[:, i], label=r'\bf{{{}}}'.format(testLabel), linewidth = 3)
        ax.plot(epochsTest,trueData[:, i], ':', label=r'\bf{{{}}}'.format(trueLabel), linewidth = 3)
        #ax.plot(epochsTest,trueData[:, i], 'o', markerfacecolor="None", markevery = 4,\
        #        label=r'\bf{{{}}}'.format(trueLabel), linewidth = 3)
        #ax.plot(epochs[ind_m],w[i,:], 'o', fillstyle='none', \
        #        label=r'\bf{Observation}', markersize = 8, markeredgewidth = 2)
        #ax.plot(epochs,ua[i,:], '--', label=r'\bf{Analysis}', linewidth = 3)
        ax.axvspan(epochsTrain[0], epochsTrain[-1], color='y', alpha=0.4, lw=0)
        #if i % 2 == 0:
        #    ax.set_ylabel(r'$z_{}(t)$'.format(i+1), labelpad=5)
        #else:
        #    ax.set_ylabel(r'$z_{}(t)$'.format(i+1), labelpad=-12)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_ylabel(r'$z_{}(t)$'.format(i+1), labelpad=5)
        i = i + 1

    axs.flat[-1].set_xlabel(r'$t$',fontsize=22)
    axs.flat[0].legend(loc="center", bbox_to_anchor=(0.5,1.25),ncol =4,fontsize=15)
    fig.subplots_adjust(hspace=0.5)

    plt.savefig(fileName, dpi = 500, bbox_inches = 'tight')
    fig.clear(True)


def subplotProbe(figNum, epochsTest, epochsTrain, trueData, testData, trainData,\
                    trueLabel, testLabel, trainLabel, fileName, px, py, var):

    fig, axs = plt.subplots(px, py, figsize=(14, 7))

    trainData = trainData.reshape(trainData.shape[0], trainData.shape[1]*trainData.shape[2])
    testData = testData.reshape(testData.shape[0], testData.shape[1]*testData.shape[2])
    trueData = trueData.reshape(trueData.shape[0], trueData.shape[1]*trueData.shape[2])

    i = 0
    for ax in axs.flat:
        ax.plot(epochsTrain,trainData[:, i], label=r'\bf{{{}}}'.format(trainLabel), linewidth = 3)
        ax.plot(epochsTest,testData[:, i], label=r'\bf{{{}}}'.format(testLabel), linewidth = 3)
        ax.plot(epochsTest,trueData[:, i], ':', label=r'\bf{{{}}}'.format(trueLabel), linewidth = 3)
        ax.axvspan(epochsTrain[0], epochsTrain[-1], color='y', alpha=0.4, lw=0)

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_ylabel(var+r'$_{}$'.format(i+1), labelpad=5)
        i = i + 1

    
    axs.flat[-1].set_xlabel(r'$t$',fontsize=22)
    axs.flat[0].legend(loc="center", bbox_to_anchor=(0.5,1.25),ncol =4,fontsize=15)
    fig.subplots_adjust(hspace=0.5)

    fig.tight_layout()
    plt.savefig(fileName, dpi = 500, bbox_inches = 'tight')
    fig.clear(True)