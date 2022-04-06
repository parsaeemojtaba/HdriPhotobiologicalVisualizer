# =============================================================================
# This module enables computing photobiological lighting properties of a scene using HDR images..
# The module also enables statistical analysis of photobiological lighting properties of the scene.. 
# The results could be plotted in false color maps.


import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.colors as colors
from matplotlib import cm

class HdriPhotobiologicalVisualizer:
    # this init method reads photometric and chromaticity calibration coefficients of the camera
    # the method also sets a location to store the false color plots
    def __init__(self, CalibrationCoeffFilePath=None, ResultFolderPath=None):
        
        if not CalibrationCoeffFilePath==None:
            if os.path.exists(CalibrationCoeffFilePath):
                self.Calibration_Coeffi_Dict = {}
                with open(CalibrationCoeffFilePath, "r") as file:
                    for line in file:
                        lines = line.split("\n", 1)
                        keys=(lines[0].split("\t", 1))[0]
                        values=(lines[0].split("\t", 1))[1]
                        self.Calibration_Coeffi_Dict[keys]=values
        else:
            print('>>> Cannot read the calibration file!')
            CalibrationCoeff_R=1
            CalibrationCoeff_G=1
            CalibrationCoeff_B=1
            CalibrationCoeff_X=1
            CalibrationCoeff_Y=1
            CalibrationCoeff_Z=1
            CalibrationCoeff_Ill=1
            CalibrationCoeff_Lum=1

            Calibration_Coefficients =  [CalibrationCoeff_R, CalibrationCoeff_G, CalibrationCoeff_B,
                                        CalibrationCoeff_X, CalibrationCoeff_Y, CalibrationCoeff_Z,
                                        CalibrationCoeff_Ill, CalibrationCoeff_Lum]
            Calibration_CoeffHeaders =  ["CalibrationCoeff_R", "CalibrationCoeff_G", "CalibrationCoeff_B",
                                        "CalibrationCoeff_X", "CalibrationCoeff_Y", "CalibrationCoeff_Z",
                                        "CalibrationCoeff_Ill", "CalibrationCoeff_Lum"]
            self.Calibration_Coeffi_Dict  =  dict(zip(Calibration_CoeffHeaders, Calibration_Coefficients))

        print(self.Calibration_Coeffi_Dict)

        if ResultFolderPath==None:
            Resultfoldername = 'Analysis_Results'
            self.ResultFolder = os.path.join(os.getcwd(), Resultfoldername)
        else:
            self.ResultFolder=ResultFolderPath

        if not os.path.exists(self.ResultFolder):
            os.makedirs(self.ResultFolder)

    # this method reads an hdr image, with any formats
    # returns a 3d array
    def reandHDRIs(self, hdrimagexPath):
        hdrimagex=cv2.imread(hdrimagexPath, cv2.IMREAD_UNCHANGED)
        return hdrimagex

    # this method extracts the calibrated RGB and XYZ chanelles, and the height and length of the HDR image
    # returns a tuple inclduing R, G, B, X, Y, Z, height and width of the HDR image after applying calibration coefficients
    def HdrChanneles(self, getimageHDR):
        x_coef = float(self.Calibration_Coeffi_Dict['CalibrationCoeff_X'])
        y_coef = float(self.Calibration_Coeffi_Dict['CalibrationCoeff_Y'])
        z_coef = float(self.Calibration_Coeffi_Dict['CalibrationCoeff_Z'])
        b_coef = float(self.Calibration_Coeffi_Dict['CalibrationCoeff_B'])
        g_coef = float(self.Calibration_Coeffi_Dict['CalibrationCoeff_G'])
        r_coef = float(self.Calibration_Coeffi_Dict['CalibrationCoeff_R'])
        Lumn_coef=float(self.Calibration_Coeffi_Dict['CalibrationCoeff_Lum'])
        Illu_coeff=float(self.Calibration_Coeffi_Dict['CalibrationCoeff_Ill'])

        ww=(getimageHDR.shape[1])
        hh=(getimageHDR.shape[0])

        getimageHDRmasked=getimageHDR
        getimageHDRmasked[getimageHDR>=999999]=np.nan

        b = getimageHDRmasked[:, :, 0]*b_coef*Lumn_coef*Illu_coeff
        g = getimageHDRmasked[:, :, 1]*g_coef*Lumn_coef*Illu_coeff
        r = getimageHDRmasked[:, :, 2]*r_coef*Lumn_coef*Illu_coeff

        X_new = (0.412453 * r + 0.357580 * g + 0.180423 * b)*x_coef
        Y_new = (0.212671 * r + 0.715160 * g + 0.072169 * b)*y_coef
        Z_new = (0.019334 * r + 0.119193 * g + 0.950227 * b)*z_coef

        r_new = (3.240479 * X_new) + (-1.53715 * Y_new) + (-0.498535 * Z_new)
        g_new = (-0.969256 * X_new) + (1.875991 * Y_new) + (0.041556 * Z_new)
        b_new = (0.055648 * X_new) + (-0.204043 * Y_new) + (1.057311 * Z_new)

        HDRI_chans = (r_new, g_new, b_new, X_new, Y_new, Z_new, hh, ww)
        return HDRI_chans

    # this method calculates photopic properties of the HDR image
    # returns a 2D array
    def Prop_Photopic(self, r_new, g_new, b_new):
        photopic_inv = (0.212671 * r_new + 0.715160 * g_new + 0.072169 * b_new)
        photopic=photopic_inv[::-1]
        return photopic

    # this method calculates photopic properties of the HDR image
    # returns a 2D array
    def Prop_Melanopic(self, r_new, g_new, b_new):
        melanopic_inv = (0.0013 * r_new + 0.3812 * g_new + 0.6175 * b_new)
        melanopic=melanopic_inv[::-1]
        return melanopic

    # this method calculates the ratio of melanopic/photopic properties of the HDR image
    # returns a 2D array
    def Prop_RatioMP(self, r_new, g_new, b_new):
        photopic = self.Prop_Photopic(r_new, g_new, b_new)
        melanopic = self.Prop_Melanopic(r_new, g_new, b_new)
        ratioMP = np.divide(melanopic, photopic) 
        return ratioMP

    # this method calculates the correlated color temperature (CCT) properties of the HDR image based on the McCamy approach
    # returns a 2D array
    def Prop_CCTMcCamy(self, X_new, Y_new, Z_new):
        sum_xyz = X_new + Y_new + Z_new
        cord_x = np.divide(X_new, sum_xyz)
        cord_y = np.divide(Y_new, sum_xyz)
        n = np.divide((cord_x - 0.3320), (0.1858 - cord_y))
        CCT_inv = (437 * n**3) + (3601 * n**2) + (6861 * n) + 5514.31
        CCT_inv[CCT_inv>50000]=50000
        CCT_inv[CCT_inv<1000]=1000
        CCTs=CCT_inv[::-1]
        return CCTs

    # this method calculates the correlated color temperature (CCT) properties of the HDR image based on the CIE-D65 approach
    # returns a 2D array
    def Prop_CCTcieD65(self, X_new, Y_new, Z_new):
        sum_xyz = X_new + Y_new + Z_new
        cord_x = np.divide(X_new, sum_xyz)
        cord_y = np.divide(Y_new, sum_xyz)
        n = np.divide((cord_x - 0.3320), (0.1858 - cord_y))
        CCT_inv = (449 * n**3) + (3525 * n**2) + (6823.3 * n) + 5518.87
        CCT_inv[CCT_inv>50000]=50000
        CCT_inv[CCT_inv<1000]=1000
        CCTs=CCT_inv[::-1]
        return CCTs

    # this method calculates the statistical information for a photobiological property of the HDR image
    # returns a dictionary
    def statistical_Info(self, dataIn):
        data_flat=dataIn.flatten()
        nan_array = np.isnan(data_flat)
        not_nan_array = ~ nan_array
        dataset = data_flat[not_nan_array]
        data_75th = np.around(np.percentile(dataset, 75),2)
        data_25th = np.around(np.percentile(dataset, 25),2)
        data_std = np.around(np.std(dataset),2)
        data_mean = np.around(np.mean(dataset),2)
        data_median = np.around(np.median(dataset),2)
        data_max = np.around(np.max(dataset),2)
        data_min = np.around(np.min(dataset),2)
        data_statsDict=dict(min=data_min, per25th=data_25th, mean=data_mean, median=data_median, 
                                per75th=data_75th, max=data_max, std=data_std,)
        return data_statsDict

    # this method plots a false color map of a photobiological property of the HDR image
    # returns a Matplotlib color mesh and color bar
    def Props_fcm(self, HDRI_chans, melanopic=False, photopic=False, MPratio=False, CCTcie=False, CCTMcCamy=False,
                            plotsize=None, cmap=None, vmin=None, vmax=None, log=False, plotColorBar=True, 
                            title=None, plot_dpi=None, Output_filename=None, Output_fileExt=None, **plt_params):
        if log==False:
            norm=colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        else:
            norm=colors.LogNorm(vmin=vmin, vmax=vmax, clip=True)

        if melanopic==True:
            Prop_Photobiologic = self.Prop_Melanopic(HDRI_chans[0], HDRI_chans[1], HDRI_chans[2])
            title='Melanopic luminance' if title==None else title
            cmap = 'nipy_spectral' if cmap==None else cmap
            Output_filename = 'fcm_Melanopic' if Output_filename==None else Output_filename
        if photopic==True:
            Prop_Photobiologic = self.Prop_Photopic(HDRI_chans[0], HDRI_chans[1], HDRI_chans[2])
            title='Photopic luminance' if title==None else title
            cmap = 'nipy_spectral' if cmap==None else cmap
            Output_filename = 'fcm_Photopic' if Output_filename==None else Output_filename
        if MPratio==True:
            Prop_Photobiologic = self.Prop_RatioMP(HDRI_chans[0], HDRI_chans[1], HDRI_chans[2])
            title='M/P ratio' if title==None else title
            cmap = 'RdBu' if cmap==None else cmap
            Output_filename = 'fcm_MPratio' if Output_filename==None else Output_filename
        if CCTcie==True:
            Prop_Photobiologic = self.Prop_CCTcieD65(HDRI_chans[3], HDRI_chans[4], HDRI_chans[5])
            title='CCT (K)' if title==None else title
            cmap = cm.get_cmap('jet').reversed() if cmap==None else cmap
            Output_filename = 'fcm_CCTsCie' if Output_filename==None else Output_filename
        if CCTMcCamy==True:
            Prop_Photobiologic = self.Prop_CCTMcCamy(HDRI_chans[3], HDRI_chans[4], HDRI_chans[5])
            title='CCT (K)' if title==None else title
            cmap = cm.get_cmap('jet').reversed() if cmap==None else cmap
            Output_filename = 'fcm_CCTsMcCamy' if Output_filename==None else Output_filename

        fig, ax = plt.subplots(constrained_layout=True, sharex=True, sharey=True)

        fcm = ax.pcolormesh(Prop_Photobiologic, norm=norm, cmap=cmap, alpha=1)
        ax.set_aspect(aspect='equal')
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().axes.get_xaxis().set_visible(False)
        ax.set_frame_on(False)
        if plotColorBar==True and not plt_params==None:
            cb=self.setColorBar(fcm, ax, plt_params, title=title)
        
        Output_fileExt='.png' if Output_fileExt==None else Output_fileExt
        output_plotfile = Output_filename + Output_fileExt
        output_plotfilepath = os.path.join(self.ResultFolder, output_plotfile)
        plot_dpi = 72 if plot_dpi==None else plot_dpi
        if not plotsize==None:
            fig.set_size_inches(plotsize)
        fig.savefig(output_plotfilepath, dpi=plot_dpi, bbox_inches = 'tight')

        return fig

    def setColorBar(self, fcm, ax, params, title=None):
        cb = plt.colorbar(fcm, ax=ax, drawedges=params['drawedges'], format=params['formatter'], extend=params['extend'],
                                    shrink=params['shrink'], aspect=params['aspect'], pad=params['cbpad'], 
                                    orientation=params['orientation'], location=params['location'], 
                                    extendrect=params['extendrect'])
        cb.ax.tick_params(which = 'major', labelsize=params['labelsize'], length=params['length'])
        cb.ax.tick_params(which = 'minor', length=params['minorticklength']) 
        cb.ax.set_title(title, pad=params['titlepad'], size=params['labelsize'])
        
        return cb
