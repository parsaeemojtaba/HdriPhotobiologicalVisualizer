# =============================================================================
# This script plots photobiological properties of a scene using the HDRI Photobiological Visualizer module 


from matplotlib.ticker import ScalarFormatter
from matplotlib import cm
from HDR_PhotobiologicalLigtingAnalysis_20220321_13h15 import HdriPhotobiologicalVisualizer
# from HdriPhotobiologicalVisualizer import HdriPhotobiologicalVisualizer

# address to an HDR image
HDRI_path = r'C:\Users\parsa\Desktop\Tests\20200102_11h08m_54\Analysis_Results\pm.hdr'

# set a location to store plots
ResultFolderPath=r'C:\Users\parsa\Desktop\Tests\20200102_11h08m_54\Analysis_Results5'

# read photometric and chromaticity calibration coefficients, if exists
CalibrationCoeffFilePath=r'C:\Users\parsa\Desktop\Tests\20200102_11h08m_54\CalibrationCoeffFile.txt'

HdriPV = HdriPhotobiologicalVisualizer(CalibrationCoeffFilePath, ResultFolderPath)

# read an HDR image
getimageHDR = HdriPV.reandHDRIs(HDRI_path)

# get R, G, B, X, Y, Z, height and width of the HDR image after applying calibration coefficients
HDRI_chans = HdriPV.HdrChanneles(getimageHDR)

prop_melanopic = HdriPV.Prop_Melanopic(HDRI_chans[0], HDRI_chans[1], HDRI_chans[2])
stat_melanopic = HdriPV.statistical_Info(prop_melanopic)
print(stat_melanopic)

prop_photopic = HdriPV.Prop_Photopic(HDRI_chans[0], HDRI_chans[1], HDRI_chans[2])
stat_photopic = HdriPV.statistical_Info(prop_photopic)
print(stat_photopic)

prop_mpratio = HdriPV.Prop_RatioMP(HDRI_chans[0], HDRI_chans[1], HDRI_chans[2])
stat_mpratio = HdriPV.statistical_Info(prop_mpratio)
print(stat_mpratio)

prop_cct = HdriPV.Prop_CCTcieD65(HDRI_chans[3], HDRI_chans[4], HDRI_chans[5])
stat_cct = HdriPV.statistical_Info(prop_cct)
print(stat_cct)

dpi_img=72
im_w=(HDRI_chans[7]/dpi_img)    # image width in inch
im_h=(HDRI_chans[6]/dpi_img)    # image height in inch

plotsize=(im_h, im_w)

plt_params = {
        # set location from {'left', 'right', 'top', 'bottom'}
        "location" : 'bottom',
        # set orientation from {'vertical', 'horizontal'}
        "orientation" : 'horizontal',
        # set the endpoint of the color bar from {'neither', 'both', 'min', 'max'} 
        "extend" :'both',
        # extendrect: If False the minimum and maximum colorbar extensions will be triangular (the default). 
        # extendrect: If True the extensions will be rectangular.
        "extendrect" : False,
        # Whether to draw lines at color boundaries
        "drawedges" : False,
        "shrink" : .8,
        "aspect" : 25,
        "labelsize" : 20,
        "length" : 10,
        "minorticklength" : 8,
        "titlepad" : 10,
        "cbpad" : .01,
        "formatter" : ScalarFormatter()
        }

vmin=10         # minimum value
vmax=1000       # maximum value
HdriPV.Props_fcm(HDRI_chans, melanopic=True, plotsize=plotsize, vmin=vmin, vmax=vmax, log=True, 
                    plotColorBar=True, Output_fileExt='.jpg', **plt_params)

HdriPV.Props_fcm(HDRI_chans, photopic=True, plotsize=plotsize, vmin=vmin, vmax=vmax, log=True, 
                    plotColorBar=True, Output_fileExt='.jpg', **plt_params)

# vmin=0.35
# vmax=1.65
HdriPV.Props_fcm(HDRI_chans, MPratio=True, plotsize=plotsize, log=False, 
                    plotColorBar=True, Output_fileExt='.jpg', **plt_params)

vmin=2000
vmax=7000
cmapx = cm.get_cmap('jet', 10)
cmap = cmapx.reversed()
HdriPV.Props_fcm(HDRI_chans, CCTcie=True, plotsize=plotsize, vmin=vmin, vmax=vmax, log=False, 
                    plotColorBar=True, Output_filename='CCTcie', cmap=cmap, Output_fileExt='.jpg', **plt_params)

HdriPV.Props_fcm(HDRI_chans, CCTMcCamy=True, plotsize=plotsize, vmin=vmin, vmax=vmax, log=False, 
                    plotColorBar=True, Output_filename='CCTMcCamy', cmap=cmap, Output_fileExt='.jpg', **plt_params)
