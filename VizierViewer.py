########################
#  VizieR Data Viewer  #
#    Steven C. Adams   #
#      Version 1.0     #
#      Jan 3, 2018     #
########################

import numpy as np
import matplotlib.pyplot as plt
import urllib2

from scipy.interpolate import interp1d

import warnings
warnings.filterwarnings("ignore")

from astropy.io.votable import parse_single_table

##########################################
#  Begin section to look at VizieR data  #
##########################################
path_sources = ''

query_info = np.genfromtxt(path_sources+'source.dat',
                           skip_header=1,
                           names=['Source','Radius'],
                           dtype=None,
                           delimiter=','
                          )

sources = query_info['Source']
radii   = query_info['Radius']

see_list = raw_input("Would you like to view your list of sources? (yes/no[DEFAULT]) ")
if see_list == "yes":
    print sources

#Define path to your VOTable data file(s).
data_path = ''
star = raw_input("Which star would you like to look at? ")

#Define data variable and save information to a table.
#THIS IS CASE SENSITIVE. MAKE SURE FILE NAME HAS PROPER CAPITALIZATION
data_orig = parse_single_table(data_path+star+".vot").to_table()

filt_RA  = data_orig['_RAJ2000'][0]
filt_DEC = data_orig['_DEJ2000'][0]

#Create array to store names [Not entirely necessary...]
names = data_orig.colnames

#Define variable from Table of data
freq_orig  = data_orig['sed_freq']   #in GHz
lam_orig   = 299792.458/freq_orig    #convert to microns
flux_orig  = data_orig['sed_flux']   #in Jansky
eflux_orig = data_orig['sed_eflux']  #ERROR in Jansky
f_lam_orig  = 3.e-9*flux_orig/(lam_orig*lam_orig)   #units of erg/s/cm2/micron
ef_lam_orig = 3.e-9*eflux_orig/(lam_orig*lam_orig)  #ERROR in erg/s/cm2/micron
nufnu_orig  = 1.e-17*flux_orig*freq_orig            #units of W/m2
enufnu_orig = 1.e-17*eflux_orig*freq_orig           #ERROR in W/m2

#Sort data by ascending Frequency
data = np.sort(data_orig, order='sed_freq')

freq  = data['sed_freq']
lam   = 299792.458/freq
flux  = data['sed_flux']
eflux = data['sed_eflux']
f_lam  = 3.e-9*flux/(lam*lam)
ef_lam = 3.e-9*eflux/(lam*lam)
nufnu  = 1.e-17*flux*freq
enufnu = 1.e-17*eflux*freq

#Any rows with Negative flux values
indexes = [index for index, value in enumerate(data_orig['sed_flux']) if value < 0]
print "{} rows with negative flux values" .format(len(indexes))

#Define plotting variables
freq_min = np.amin(freq)
freq_max = np.amax(freq)
lam_min  = np.amin(lam)
lam_max  = np.amax(lam)

######################################
#  Section for plotting VizieR Data  #
######################################
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

###########################################
#  Remove ROWS with negative flux values  #
###########################################

test = data_orig

indexes = [index for index, value in enumerate(data_orig['sed_flux']) if value < 0]
for index in indexes:
    print(index)  #Prints rows that have negative flux values

test = np.delete(test, (indexes), axis=0)    

print "Original table length {}" .format(len(data_orig))  #To check and see if all
print "New table length {}" .format(len(test))            #rows were removed


################################################
#  Section to fit VizieR data (linear interp)  #
################################################

#Sort new table with negative fluxes removed
#Sort data by ascending Frequency
#test2 = np.sort(test, order='sed_freq')

freq_t   = test['sed_freq']
lam_t    = 299792.458/freq_t
flux_t   = test['sed_flux']
eflux_t  = test['sed_eflux']
f_lam_t  = 3.e-9*flux_t/(lam_t*lam_t)
ef_lam_t = 3.e-9*eflux_t/(lam_t*lam_t)
nufnu_t  = 1.e-17*flux_t*freq_t
enufnu_t = 1.e-17*eflux_t*freq_t

#To ensure delta lambda is consistent for each source.
#Edit denominator below to desired delta lambda.
dlam = (max(lam_t) - min(lam_t))/0.005

lamnew  = np.linspace(lam_min, lam_max, num=dlam, endpoint=True)
#freqnew = np.linspace(freq_min, freq_max, num=20000, endpoint=True) 
freqnew = 299792.458/lamnew

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

#############################################
#  Pick Wavelength range to determine Flux  #
#############################################

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

#################################################################
#  Plot and print interpolated Flux values in Wavelength Range  #
#################################################################

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

show_tab = raw_input("Do you want to show flux value table based on fit?[yes/no[DEFAULT] ")
if show_tab == "yes":
    print "\n\n   Microns   ","  Freq (GHz) ", "   Jansky   ", "  erg/s/cm2/um  ", "   W/m2    "
    indexes = [index for index, value in enumerate(lamnew) if value < max_lam and value > min_lam]
    for index in indexes:
        print '{:1.7e} {:1.7e} {:1.7e} {:1.8e} {:1.7e}'.format(c1[index], c2[index], c3[index], c4[index], c5[index])

#######################################
#  To be used with SEDFITTER Package  #
#######################################

#To be used for fitter SED with Kurucz model.
#See here for more information on SEDFITTER:
# http://sedfitter.readthedocs.io/en/stable/introduction.html

filters = test['sed_filter']
i=0

filt_list = set(filters)
temp = list(filt_list)
unique_filts = sorted(temp)

avg_flux  = []
avg_freq  = []
avg_eflux = []

for index in unique_filts:
    flux_temp  = 0.
    freq_temp  = 0.
    eflux_temp = 0.
    j = 0.
    for i, name in enumerate(test['sed_filter']):
        if name == index:
            flux_temp = flux_temp + test['sed_flux'][i]
            freq_temp = freq_temp + test['sed_freq'][i]
            if np.isnan(test['sed_eflux'][i]) == True:
                test['sed_eflux'][i] = 0.
            eflux_temp = eflux_temp + test['sed_eflux'][i]
            j = j+1.
    avg_flux.append((flux_temp/j)*1000.)
    avg_freq.append(freq_temp/j)
    avg_eflux.append((eflux_temp/j)*1000.)

#Used to write to file for SED flux values
known_filt = ['2MASS:H','2MASS:J','2MASS:Ks','IRAS:100','IRAS:12','IRAS:25','IRAS:60','WISE:W1','WISE:W2','WISE:W3','WISE:W4']
filt = ['2H','2J','2K','S4','S1','S2','S3','WISE1','WISE2','WISE3','WISE4']

fitter_flux = []
fitter_err  = []
fitter_flag = []

for i, name in enumerate(known_filt):
    for index in unique_filts:
        if name == index:
            fitter_flux.append(avg_flux[i])
            fitter_err.append(avg_eflux[i])
            fitter_flag.append(1)

filename=star+".dat"

temp = open(filename,"w")

temp.write(star)
temp.write(" ")
temp.write(str(filt_RA))
temp.write(" ")
temp.write(str(filt_DEC))
temp.write(" ")

for j in fitter_flag:
    temp.write(str(fitter_flag[j]))
    temp.write(" ")

indexes = [index for index, value in enumerate(fitter_flux)]
for i in indexes:
    temp.write(str(fitter_flux[i]))
    temp.write(" ")
    temp.write(str(fitter_err[i]))
    temp.write(" ")

temp.close()
