###########################
#  VizieR Data Collector  #
#     Steven C. Adams     #
#       Version 1.0       #
#        Jan 3, 2018      #
###########################

#Run Before Vizier Data Viewer

import numpy as np
import matplotlib.pyplot as plt
import urllib2

from scipy.interpolate import interp1d

from astropy.io.votable import parse_single_table

#############################################
#        Download VOTable files here        #
#############################################

#If you run this code with the same source list
#it will overwrite any files with the same name
#as any previous runs.

#############################################
#              IMPORTANT: READ              #
#############################################
# Edit sources in this list. 
# Must be strings.
# Prefereably, no spaces.
# If spaces are needed, replace [space] with %20 for correct URL
# Case does not matter.

path_sources = ''

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
    print "Sources length = {} : Radii length = {}" .format(len(sources),len(radii))

#Can also edit search radius.
#Create new list with radius for each source
#Default is 5 arcsec
for i, source in enumerate(sources):
    response = urllib2.urlopen('http://vizier.u-strasbg.fr/viz-bin/sed?-c='+source+'&-c.rs='+str(radii[i]))
    html = response.read()
    with open(source+".vot", "wb") as code:
        code.write(html)                    #Writes out html string to VOTable format
