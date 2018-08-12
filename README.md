# VizierGrabber
Download, Plot, and Investigate VizieR Photometry Data

This program is used to download different sources' photometry data 
from the VizieR photometry viewer. It also fits a 1D interpolation 
to the data in order to determine estimates of flux density at points 
not covered with photometric data.

Steven C. Adams
Clemson University

Four files are necessary to run this program. 

The source.dat file is for the sources you are wanting to download. The format is [sourcename],[search radius] with no spaces. If you need spaces for the name, replace the space with %20 so the URL will be found for the download.

The next file is VizierGrabber.ipynb. This is the Python notebook file that you can upload and run on Jupyter Hub. It is basically the same as the .py files I have added also, just in a format that is easier to edit yourself if you have something that can format the Python Notebook file.

If you want to run this on your own machine, you will need the SciPy, NumPy, AstroPy, and MatPlotLib libraries installed. The first file (VizierCollector.py) is the download program. This currently just places the downloaded .vot files in your working directory. This may take a few seconds, depending on the number of files you're downloading (for 3 files, the most it has taken for me is maybe 10 seconds). Run this first.

The next file is the VizierViewer.py. This is the file that converts the .vot file. You can select the x and y variables via input. Make sure you use the exact name as shown in the square brackets, case and parentheses. I still have the 1D interpolation to fit the data, but I am working on the SED fitter program to fit the data, just having issues with that...
