import uproot
import awkward
import pandas as pd
import pickle
from array import array
import sys,os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FuncFormatter
import boost_histogram as bh
from plothist import make_hist, plot_hist, plot_error_hist
from fetchscalers_coin import fetchscalers_coin
from fetchnormfac import fetchnormfac
from makedathistos import makedathistos

# set up plotting defaults
size=15
params = {'legend.fontsize': size,
          'font.weight':'normal',
          'figure.figsize': (8.5,8.5), #size of figure
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size,
          'ytick.labelsize': size,
          'font.size':size,
          'axes.titlepad': size,
          'axes.linewidth': 1,
          'lines.linewidth': 1,
          'mathtext.default': 'regular'}
plt.rcParams.update(params)

def setup_ticks(ax):
    ax.tick_params(which="both",direction="in")
    ax.tick_params(which='major', width=1.0)
    ax.tick_params(which='major', length=10)
    ax.tick_params(which='minor', width=1.0, )
    ax.tick_params(which='minor', length=5, )
    ax.tick_params(which='both', right=True, top=True)
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))


#Inputs - I need to simplify this    
kin= 'rsidis_x0p25_q23p3_z0p5_pip_hyd_loweps'   
simfile='rsidis_x0p25_q23p3_z0p5_pip_hyd_loweps'
mytype='PI+SIDIS'
#mybeam=10.67
mybeam=8.565
mymom=3.632
myhmom=-1.531
mytheta=7.87
#myhmom=-3.642
#mymom=-4.868
#mytheta=10.305
mytarg='LH2'

## read in runlist##
filename = "rsidis_runlist.dat"
if (os.path.isfile(filename)):
    target,runtype =np.loadtxt(filename,unpack=True,comments='!',usecols=(5,11),dtype='str')
    runno,ebeam,phms,th_hms,pshms,th_shms = np.loadtxt(filename,unpack=True,comments='!', usecols=(0,3,6,7,8,9),dtype='float')

cut = (mytarg==target) & (mytype==runtype) & (mymom==pshms) & (mytheta==th_shms) & (mybeam==ebeam) & (myhmom==phms)
dataruns=runno[cut]

doing_cryo=False
if mytarg=='LH2' or mytarg=='LD2':
    doing_cryo=True

if doing_cryo:    
    dumtarg='Dummy'
    cut = (dumtarg==target) & (mytype==runtype) & (mymom==pshms) & (mytheta==th_shms) & (mybeam==ebeam) & (myhmom==phms)
    dummyruns=np.array(runno[cut])
#    dummyruns=np.array([24335])

print('Data runs: ',dataruns)
if doing_cryo:
    print('Dummy runs: ',dummyruns)

# coin time histos
h2001 = bh.Histogram(bh.axis.Regular(bins=50, start=-20.0, stop=4.0))
h2002 = bh.Histogram(bh.axis.Regular(bins=50, start=-20.0, stop=4.0))
h2003 = bh.Histogram(bh.axis.Regular(bins=50, start=-20.0, stop=4.0))

nvar=14
histos={}
for i in range(nvar):
    histos[i]={}

nhist=13

for i in range(nhist):
    # HMS reconstructed quantities
    histos[0,i] = bh.Histogram(bh.axis.Regular(bins=16, start=-8.0, stop=8.0), storage=bh.storage.Weight()) # HMS delta
    histos[1,i] = bh.Histogram(bh.axis.Regular(bins=50, start=-0.1, stop=0.1), storage=bh.storage.Weight()) # HMS xptar   
    histos[2,i] = bh.Histogram(bh.axis.Regular(bins=50, start=-0.06, stop=0.06), storage=bh.storage.Weight()) # HMS yptar
    histos[3,i] = bh.Histogram(bh.axis.Regular(bins=50, start=-5.0, stop=5.0), storage=bh.storage.Weight()) # HMS ytar
    #SHMS reconstructed quantities
    histos[4,i] = bh.Histogram(bh.axis.Regular(bins=30, start=-10.0, stop=20.0), storage=bh.storage.Weight()) # SHMS delta
    histos[5,i] = bh.Histogram(bh.axis.Regular(bins=50, start=-0.1, stop=0.1), storage=bh.storage.Weight()) # SHMS xptar   
    histos[6,i] = bh.Histogram(bh.axis.Regular(bins=50, start=-0.06, stop=0.06), storage=bh.storage.Weight()) # SHMS yptar
    histos[7,i] = bh.Histogram(bh.axis.Regular(bins=50, start=-5.0, stop=5.0), storage=bh.storage.Weight()) # SHMS ytar
    # Physics things
    histos[8,i] = bh.Histogram(bh.axis.Regular(bins=50, start=2.0, stop=4.0), storage=bh.storage.Weight()) # W
    histos[9,i] = bh.Histogram(bh.axis.Regular(bins=50, start=1.0, stop=4.0), storage=bh.storage.Weight()) # Q2   
    histos[10,i] = bh.Histogram(bh.axis.Regular(bins=50, start=0.1, stop=0.4), storage=bh.storage.Weight()) # xbj
    histos[11,i] = bh.Histogram(bh.axis.Regular(bins=45, start=0.1, stop=1.0), storage=bh.storage.Weight()) # z
    histos[12,i] = bh.Histogram(bh.axis.Regular(bins=50, start=0.0, stop=0.5), storage=bh.storage.Weight()) # Pt
    histos[13,i] = bh.Histogram(bh.axis.Regular(bins=50, start=0.0, stop=360.0), storage=bh.storage.Weight()) # Phi
    

#####Data runs ######
qtot_data = 0
qtot_data = makedathistos(dataruns,1,histos,nvar,1.0,kin)

#####Dummy runs if cryo target######
if doing_cryo:
    if mytarg=='LH2':
        dumrat=3.550
    elif mytarg=='LD2':
        dumrat=3.7825
        
    qtot_dummy = 0
    qtot_dummy = makedathistos(dummyruns,6,histos,nvar,dumrat,kin)


#subtract dummy from the data
for i in range(nvar):
    histos[i,11]=histos[i,3]+histos[i,8]*(-1.0)

#now make the simc histograms
(normfac, ngen) = fetchnormfac(simfile)
simtree='worksim/'+simfile+'.root:h10'
print('Opening simc file ', simtree)
f=uproot.open(simtree)


hsdelta = f['hsdelta'].array(library="np") #open each value we want into a numpy array
hsyptar = f['hsyptar'].array(library="np") 
hsxptar = f['hsxptar'].array(library="np") 
hsytar = f['hsytar'].array(library="np") 
psdelta = f['ssdelta'].array(library="np")
psyptar = f['ssyptar'].array(library="np")
psxptar = f['ssxptar'].array(library="np")
psytar = f['ssytar'].array(library="np")
W = f['W'].array(library="np")
Q2 = f['Q2'].array(library="np")
xbj = f['xbj'].array(library="np")
zhad = f['z'].array(library="np")
pt2 = f['pt2'].array(library="np")
phipq = f['phipq'].array(library="np")
mcweight = f['Weight'].array(library="np")

df = pd.DataFrame({'hsdelta': hsdelta, 'hsxptar': hsxptar, 'hsyptar':hsyptar, 'hsytar':hsytar, 'psdelta': psdelta, 'psyptar': psyptar,  'psxptar': psxptar,  'psytar': psytar, 'W': W, 'Q2': Q2, 'xbj': xbj, 'zhad': zhad, 'pt2': pt2, 'phipq': phipq, 'mcweight':mcweight})

df['pt'] = np.sqrt(df['pt2']) 

df_mccut = df[abs(df['hsdelta'] < 8) & (df['psdelta']>-10.0) & (df['psdelta']<20.0)]


simweight= (normfac/ngen)*df_mccut['mcweight']

histos[0,12].fill(df_mccut['hsdelta'],weight=simweight)
histos[1,12].fill((df_mccut['hsxptar']),weight=simweight)
histos[2,12].fill((df_mccut['hsyptar']),weight=simweight)
histos[3,12].fill((df_mccut['hsytar']),weight=simweight)
histos[4,12].fill(df_mccut['psdelta'],weight=simweight)
histos[5,12].fill((df_mccut['psxptar']),weight=simweight)
histos[6,12].fill((df_mccut['psyptar']),weight=simweight)
histos[7,12].fill((df_mccut['psytar']),weight=simweight)
histos[8,12].fill((df_mccut['W']),weight=simweight)
histos[9,12].fill((df_mccut['Q2']),weight=simweight)
histos[10,12].fill((df_mccut['xbj']),weight=simweight)
histos[11,12].fill((df_mccut['zhad']),weight=simweight)
histos[12,12].fill((df_mccut['pt']),weight=simweight)
histos[13,12].fill((df_mccut['phipq']),weight=simweight)

#dump the histos
picklefile='HISTOS/'+kin+'_histos.pkl'
with open(picklefile, "wb") as f:
    pickle.dump(histos, f)

### do the plotting
plotfile='PLOTS/'+kin+'.pdf'
with PdfPages(plotfile) as pdf:
#plt.figure(1)
#ax0=plt.subplot(111)
#plot_hist(h2003, ax=ax0, histtype="step", color='black',linewidth=1.2, label="c1")
#plot_hist(h2001, ax=ax0, histtype="step", color='red', linewidth=1.2, label="c1")
#plot_hist(h2002, ax=ax0, histtype="step", color='blue', linewidth=1.2, label="c1")


    
    plt.figure(1)
    ax1=plt.subplot(221)
    ax1.set_xlabel('HMS delta')
    ax1.set_ylabel('Yield (counts/mC)')
    plot_error_hist(histos[0,3], ax=ax1, color='black',linewidth=1.2, label="Data")
    plot_error_hist(histos[0,8], ax=ax1, color='red', linewidth=1.2, label="Dummy")
    plot_error_hist(histos[0,11], ax=ax1, color='blue', linewidth=1.2, label="Data-dummy")
    plot_hist(histos[0,12], ax=ax1, histtype="step",color='blue', linewidth=1.2, label="SIMC")
    ax1.set_ylim(bottom=0)
    plt.tight_layout()

    ax2=plt.subplot(222)
    ax2.set_xlabel('HMS xptar')
    ax2.set_ylabel('Yield (counts/mC)')
    plot_error_hist(histos[1,3], ax=ax2, color='black',linewidth=1.2)
    plot_error_hist(histos[1,8], ax=ax2, color='red', linewidth=1.2)
    plot_error_hist(histos[1,11], ax=ax2, color='blue', linewidth=1.2)
    plot_hist(histos[1,12], ax=ax2, histtype="step",color='blue', linewidth=1.2, label="SIMC")
    ax2.set_ylim(bottom=0)
    plt.tight_layout()

    ax3=plt.subplot(223)
    ax3.set_xlabel('HMS yptar')
    ax3.set_ylabel('Yield (counts/mC)')
    plot_error_hist(histos[2,3], ax=ax3, color='black',linewidth=1.2)
    plot_error_hist(histos[2,8], ax=ax3, color='red', linewidth=1.2)
    plot_error_hist(histos[2,11], ax=ax3, color='blue', linewidth=1.2)
    plot_hist(histos[2,12], ax=ax3, histtype="step",color='blue', linewidth=1.2, label="SIMC")
    ax3.set_ylim(bottom=0)
    plt.tight_layout()

    ax4=plt.subplot(224)
    ax4.set_xlabel('HMS ytar')
    ax4.set_ylabel('Yield (counts/mC)')
    plot_error_hist(histos[3,3], ax=ax4, color='black',linewidth=1.2)
    plot_error_hist(histos[3,8], ax=ax4, color='red', linewidth=1.2)
    plot_error_hist(histos[3,11], ax=ax4, color='blue', linewidth=1.2)
    plot_hist(histos[3,12], ax=ax4, histtype="step",color='blue', linewidth=1.2, label="SIMC")
    ax4.set_ylim(bottom=0)
    plt.tight_layout()
    pdf.savefig()
    
    plt.figure(2)
    ax1=plt.subplot(221)
    ax1.set_xlabel('SHMS delta')
    ax1.set_ylabel('Yield (counts/mC)')
    plot_error_hist(histos[4,3], ax=ax1, color='black',linewidth=1.2, label="Data")
    plot_error_hist(histos[4,8], ax=ax1, color='red', linewidth=1.2, label="Dummy")
    plot_error_hist(histos[4,11], ax=ax1, color='blue', linewidth=1.2, label="Data-dummy")
    plot_hist(histos[4,12], ax=ax1, histtype="step",color='blue', linewidth=1.2, label="SIMC")
    ax1.set_ylim(bottom=0)
    plt.tight_layout()

    ax2=plt.subplot(222)
    ax2.set_xlabel('SHMS xptar')
    ax2.set_ylabel('Yield (counts/mC)')
    plot_error_hist(histos[5,3], ax=ax2, color='black',linewidth=1.2)
    plot_error_hist(histos[5,8], ax=ax2, color='red', linewidth=1.2)
    plot_error_hist(histos[5,11], ax=ax2, color='blue', linewidth=1.2)
    plot_hist(histos[5,12], ax=ax2, histtype="step",color='blue', linewidth=1.2, label="SIMC")
    ax2.set_ylim(bottom=0)
    plt.tight_layout()
    
    ax3=plt.subplot(223)
    ax3.set_xlabel('SHMS yptar')
    ax3.set_ylabel('Yield (counts/mC)')
    plot_error_hist(histos[6,3], ax=ax3, color='black',linewidth=1.2)
    plot_error_hist(histos[6,8], ax=ax3, color='red', linewidth=1.2)
    plot_error_hist(histos[6,11], ax=ax3, color='blue', linewidth=1.2)
    plot_hist(histos[6,12], ax=ax3, histtype="step",color='blue', linewidth=1.2, label="SIMC")
    ax3.set_ylim(bottom=0)
    plt.tight_layout()

    ax4=plt.subplot(224)
    ax4.set_xlabel('SHMS ytar')
    ax4.set_ylabel('Yield (counts/mC)')
    plot_error_hist(histos[7,3], ax=ax4, color='black',linewidth=1.2)
    plot_error_hist(histos[7,8], ax=ax4, color='red', linewidth=1.2)
    plot_error_hist(histos[7,11], ax=ax4, color='blue', linewidth=1.2)
    plot_hist(histos[7,12], ax=ax4, histtype="step",color='blue', linewidth=1.2, label="SIMC")
    ax4.set_ylim(bottom=0)
    plt.tight_layout()
    
    pdf.savefig()


    plt.figure(3)
    ax1=plt.subplot(221)
    ax1.set_xlabel('W')
    ax1.set_ylabel('Yield (counts/mC)')
    plot_error_hist(histos[8,3], ax=ax1, color='black',linewidth=1.2, label="Data")
    plot_error_hist(histos[8,8], ax=ax1, color='red', linewidth=1.2, label="Dummy")
    plot_error_hist(histos[8,11], ax=ax1, color='blue', linewidth=1.2, label="Data-dummy")
    plot_hist(histos[8,12], ax=ax1, histtype="step",color='blue', linewidth=1.2, label="SIMC")
    ax1.set_ylim(bottom=0)
    plt.tight_layout()

    ax2=plt.subplot(222)
    ax2.set_xlabel('Q2')
    ax2.set_ylabel('Yield (counts/mC)')
    plot_error_hist(histos[9,3], ax=ax2, color='black',linewidth=1.2)
    plot_error_hist(histos[9,8], ax=ax2, color='red', linewidth=1.2)
    plot_error_hist(histos[9,11], ax=ax2, color='blue', linewidth=1.2)
    plot_hist(histos[9,12], ax=ax2, histtype="step",color='blue', linewidth=1.2, label="SIMC")
    ax2.set_ylim(bottom=0)
    plt.tight_layout()
    
    ax3=plt.subplot(223)
    ax3.set_xlabel('xBj')
    ax3.set_ylabel('Yield (counts/mC)')
    plot_error_hist(histos[10,3], ax=ax3, color='black',linewidth=1.2)
    plot_error_hist(histos[10,8], ax=ax3, color='red', linewidth=1.2)
    plot_error_hist(histos[10,11], ax=ax3, color='blue', linewidth=1.2)
    plot_hist(histos[10,12], ax=ax3, histtype="step",color='blue', linewidth=1.2, label="SIMC")
    ax3.set_ylim(bottom=0)
    plt.tight_layout()

    ax4=plt.subplot(224)
    ax4.set_xlabel('Zhad')
    ax4.set_ylabel('Yield (counts/mC)')
    plot_error_hist(histos[11,3], ax=ax4, color='black',linewidth=1.2)
    plot_error_hist(histos[11,8], ax=ax4, color='red', linewidth=1.2)
    plot_error_hist(histos[11,11], ax=ax4, color='blue', linewidth=1.2)
    plot_hist(histos[11,12], ax=ax4, histtype="step",color='blue', linewidth=1.2, label="SIMC")
    ax4.set_ylim(bottom=0)
    plt.tight_layout()
    
    pdf.savefig()
#calculate data and sim yield
Ydata=histos[0,11].sum()
Ydata_vec=histos[0,11].counts()
eYdata_vec = histos[0,11].variances()
Ysim=histos[0,12].sum()
Ysim_vec=histos[0,12].counts()
eYsim_vec = histos[0,12].variances()


print(f"Data yield (counts/mC): {sum(Ydata_vec):.2f} +/- {np.sqrt(sum(eYdata_vec)):.2f}")
print(f"SIMC yield (counts/mC): {sum(Ysim_vec):.2f} +/- {np.sqrt(sum(eYsim_vec)):.2f}")

plt.show()
