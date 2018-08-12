########################
#  VizieR Data Fitter  #
#    Steven C. Adams   #
#      Version 1.0     #
#      Jan 3, 2018     #
########################

#This program maske use of the SEDFITTER package 
#for Python. See here for mor information:
#  http://sedfitter.readthedocs.io/en/stable/introduction.html

from astropy import units as u
from sedfitter import fit
from sedfitter.extinction import Extinction

# Define path to models
model_dir = '/Users/cadeadams/Desktop/PythonPrac/models_kurucz'

# Read in extinction law)
extinction = Extinction.from_file('kmh94.par', columns=[0, 3],
                                  wav_unit=u.micron, chi_unit=u.cm**2 / u.g)

# Define filters and apertures
filt = ['2H','2J','2K','S4','S1','S2','S3','WISE1','WISE2','WISE3','WISE4']
apertures = [3., 3., 3., 3., 3., 3., 3.,3.,3.,3.,3.] * u.arcsec

# Run the fitting
fit('V380Ori.dat', filt, apertures, model_dir,
    'output.fitinfo',
    extinction_law=extinction,
    distance_range=[1., 2.] * u.kpc,
    av_range=[0., 40.])

from sedfitter import plot
plot('output.fitinfo', 'plots_seds')

