
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import urllib2

from scipy.interpolate import interp1d

from astropy.io.votable import parse_single_table

import warnings
warnings.filterwarnings("ignore")

#from IPython.core.display import HTML
#HTML("<style>.container { width:98% !important; }</style>")


# In[2]:

#Download VOTable files here

#If you run this code with the same source list
#it will overwrite any files with the same name
#as any previous runs.

#Edit sources in this list. 
#Must be strings.
#Prefereably, no spaces.
#If spaces are needed, replace [space] with %20for correct URL
#Case does not matter.
path_sources = ''    #Path to the location of your source.dat file

query_info = np.genfromtxt(path_sources+'source.dat',
                           skip_header=1,
                           names=['Source','Radius'],
                           dtype=None,
                           delimiter=','
                          )

sources = query_info['Source']
radii   = query_info['Radius']

if len(sources) != len(radii):
    print "Check your sources and radii list to ensure each have the same number of elements."
    print "Sources length = {} ; Radii length = {}" .format(len(sources),len(radii))

#Can also edit search radius.
#Create new list with radius for each source
#Default is 5 arcseec
for i, source in enumerate(sources):
    response = urllib2.urlopen('http://vizier.u-strasbg.fr/viz-bin/sed?-c='+source+'&-c.rs='+str(radii[i]))
    html = response.read()
    with open(source+".vot", "wb") as code:
        code.write(html)                    #Writes out html string to VOTable format
        


# In[3]:

#Define path to your VOTable data file(s).
data_path = ''

#Define data variable and save information to a table.
#THIS IS CASE SENSITIVE. MAKE SURE FILE NAME HAS PROPER CAPITALIZATION
data_orig = parse_single_table(data_path+"HD101412.vot").to_table()

#Create array to store names [Not entirely necessary...]
names = data_orig.colnames

#Define variable from Table of data
freq_orig   = data_orig['sed_freq']                 #in GHz
lam_orig    = 299792.458/freq_orig                  #convert to microns
flux_orig   = data_orig['sed_flux']                 #in Jansky
eflux_orig  = data_orig['sed_eflux']                #ERROR in Jansky
f_lam_orig  = 3.e-9*flux_orig/(lam_orig*lam_orig)   #units of erg/s/cm2/micron
ef_lam_orig = 3.e-9*eflux_orig/(lam_orig*lam_orig)  #ERROR in erg/s/cm2/micron
nufnu_orig  = 1.e-17*flux_orig*freq_orig            #units of W/m2
enufnu_orig = 1.e-17*eflux_orig*freq_orig           #ERROR in W/m2

#Sort data by ascending Frequency (for interpolation)
#Units listed above.
data = np.sort(data_orig, order='sed_freq')

freq   = data['sed_freq']
lam    = 299792.458/freq 
flux   = data['sed_flux']
f_lam  = 3.e-9*flux/(lam*lam)
nufnu  = 1.e-17*flux*freq
eflux  = data['sed_eflux']
ef_lam = 3.e-9*eflux/(lam*lam)
enufnu = 1.e-17*eflux*freq

#Any rows with Negative flux values
indexes = [index for index, value in enumerate(data_orig['sed_flux']) if value < 0]
print "{} rows with negative flux values" .format(len(indexes))


#Define plotting variables
freq_min = np.amin(freq_orig)
freq_max = np.amax(freq_orig)
lam_min  = np.amin(lam_orig)
lam_max  = np.amax(lam_orig)


# In[4]:

xax = raw_input("X-axis Variable (Wavelength or Frequency)? [Wavelength is default] ")
if xax == 'Wavelength':
    x = lam
    xlab = r'Wavelength ($\mu$m)'
elif xax == 'Frequency':
    x = freq
    xlab = r'Frequency (GHz)'
else:
    x = lam
    xlab = r'Wavelength ($\mu$m)'

yax = raw_input("Y-axis Variable (Jansky, F(lam), nuF(nu))? [Jansky is default] ")
if yax == 'Jansky':
    y = flux
    yerr = eflux
    ylab = r'F($\nu$) (Jy)'
elif yax == 'F(lam)':
    y = f_lam
    yerr = ef_lam
    ylab = r'F($\lambda$) (erg/s/cm$^2$/$\mu$m)'
elif yax == 'nuF(nu)':
    y = nufnu
    yerr = enufnu
    ylab = r'$\nu$ F($\nu$ ) (W/m$^2$)'
else:
    y = flux
    yerr = eflux
    ylab = r'F($\nu$) (Jy)'


plt.errorbar(x,y,yerr=yerr,fmt='ro')
#plt.plot(lam,flux,'ro')   #Plot without error bars

#plt.axis([lam_min/10., lam_max*10., 0.01, 1000.])
plt.xlabel(xlab)   #Change based on plotting variables
plt.xscale('log')
plt.ylabel(ylab)             #Change based on plotting variables
plt.yscale('log')

plt.show()


# In[5]:

#Remove ROWS with negative flux values

test = data_orig

indexes = [index for index, value in enumerate(data_orig['sed_flux']) if value < 0]
for index in indexes:
    print(index)  #Prints rows that have negative flux values

test = np.delete(test, (indexes), axis=0)    

print "Original table length {}" .format(len(data_orig))  #To check and see if all
print "New table length {}" .format(len(test))            #rows were removed


# In[6]:

#Sort new table with negative fluxes removed
#Sort data by ascending Frequency
#test2 = np.sort(test, order='sed_freq')

lamnew  = np.linspace(lam_min, lam_max, num=20000, endpoint=True)
#freqnew = np.linspace(freq_min, freq_max, num=20000, endpoint=True) 
freqnew = 299792.458/lamnew

freq_t   = test['sed_freq']
lam_t    = 299792.458/freq_t
flux_t   = test['sed_flux']
eflux_t  = test['sed_eflux']
f_lam_t  = 3.e-9*flux_t/(lam_t*lam_t)
ef_lam_t = 3.e-9*eflux_t/(lam_t*lam_t)
nufnu_t  = 1.e-17*flux_t*freq_t
enufnu_t = 1.e-17*eflux_t*freq_t

if xax == 'Wavelength':
    x_f   = lam_t
    xlab  = r'Wavelength ($\mu$m)'
    x_mod = lamnew
elif xax == 'Frequency':
    x_f   = freq_t
    xlab  = r'Frequency (GHz)'
    x_mod = freqnew
else:
    x_f = lam_t
    xlab  = r'Wavelength ($\mu$m)'
    x_mod = lamnew

if yax == 'Jansky':
    y_f = flux_t
    yerr_f = eflux_t
    ylab = r'F($\nu$) (Jy)'
elif yax == 'F(lam)':
    y_f = f_lam_t
    yerr_f = ef_lam_t
    ylab = r'F($\lambda$) (erg/s/cm$^2$/$\mu$m)'
elif yax == 'nuF(nu)':
    y_f = nufnu_t
    yerr_f = enufnu_t
    ylab = r'$\nu$ F($\nu$ ) (W/m$^2$)'
else:
    y_f = flux_t
    yerr_f = eflux_t
    ylab = r'F($\nu$) (Jy)'


#Fitting a function to the data to interpolate wavenlengths missing from photometry results
#Initial vunction is a simple linear interpolator
fit = interp1d(x_f,y_f)

#Plot function
plt.errorbar(x_f,y_f,yerr=yerr_f,fmt='ro')  #Uncomment to show errorbars
plt.plot(x_mod, fit(x_mod), 'g-')         #with fit to data
#plt.plot(x_f, y_f, 'ro', x_mod, fit(x_mod), 'g-')
plt.xlabel(xlab)
plt.xscale('log')
plt.ylabel(ylab)
plt.yscale('log')
plt.show()


# In[7]:

min_lam_STR = raw_input("Input minimum wavelength (in microns) ")
min_lam = float(min_lam_STR)
max_freq = 299792.458/min_lam
max_lam_STR = raw_input("Input maximum wavelength (in microns) ")
max_lam = float(max_lam_STR)
min_freq = 299792.458/max_lam

diff = max_lam - min_lam

low  = min_lam - lam_min  #Testing to make sure input values
high = lam_max - max_lam  #are within the bounds of the data

if diff < 0:
    print "Max wavelength is smaller than min wavelength"
    min_lam = None
    max_lam = None

if low < 0:
    print "Minimum Wavelength is below data range"
    min_lam = None
    max_lam = None

if high < 0:
    print "Maximum Wavelength is above data range"
    min_lam = None
    max_lam = None


# In[8]:

plt.errorbar(x_f,y_f,yerr=yerr_f,fmt='ro')
plt.plot(x_mod, fit(x_mod), 'g-')

if xax == 'Wavelength':
    plt.xlim(min_lam, max_lam)
elif xax == 'Frequency':
    plt.xlim(min_freq, max_freq)
else:
    plt.xlim(min_lam, max_lam)

plt.xlabel(xlab)             #Change based on plotting variables
#plt.xscale('log')
plt.ylabel(ylab)             #Change based on plotting variables
plt.yscale('log')
plt.show()


# In[9]:

print "   Microns   ","  Freq (GHz) ", "   Jansky   ", "  erg/s/cm2/um  ", "   W/m2    "

if xax == 'Wavelength':
    c1 = x_mod
    c2 = 299792.458/x_mod
elif xax == 'Frequency':
    c1 = 299792.458/x_mod
    c2 = x_mod
else:
    c1 = x_mod
    c2 = 299792.458/x_mod

if yax == 'Jansky':
    c3 = fit(x_mod)
    c4 = 3.e-9*fit(x_mod)/(lamnew*lamnew)
    c5 = 1.e-17*fit(x_mod)*(299792.458/lamnew)
elif yax == 'F(lam)':
    c3 = fit(x_mod)*(lamnew*lamnew)/3.e-9
    c4 = fit(x_mod)
    c5 = freqnew*(lamnew*lamnew)*fit(x_mod)/3.e8
elif yax == 'nuF(nu)':
    c3 = 1.e17*fit(x_mod)/freqnew
    c4 = 3.e8*fit(x_mod)/(freqnew*lamnew*lamnew)
    c5 = fit(x_mod)
else:
    c3 = fit(x_mod)
    c4 = 3.e-9*fit(x_mod)/(lamnew*lamnew)
    c5 = 1.e-17*fit(x_mod)*(299792.458/lamnew)

indexes = [index for index, value in enumerate(lamnew) if value < max_lam and value > min_lam]
for index in indexes:
    print '{:1.7e} {:1.7e} {:1.7e} {:1.8e} {:1.7e}'.format(c1[index], c2[index], c3[index], c4[index], c5[index])

