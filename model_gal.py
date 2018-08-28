from __future__ import print_function
import os
from os import path
import glob
import numpy as np
from scipy import ndimage,fftpack
from astropy.io import fits
import PPXF.ppxf_util as util
import pdb
from PPXF.cap_readcol import readcol
from PPXF.miles_util import miles
import bc03_util as lib
import scipy.integrate as integrate
import pdb
import matplotlib.pyplot as plt
import emission_line_tpl as emission

def exp_SFH(tage,tau=1.,t0=13.7,norm=1.):
    tt=13.7-tage
    if tt < 0:
        tt=0.
    #pdb.set_trace()
    SFR=norm*np.exp(-1.*tt/tau)
    return SFR

def dexp_SFH(tage,tau=1.,t0=13.7,norm=1.):
    tt=13.7-tage
    if tt < 0:
        tt=0.
    #pdb.set_trace()
    SFR=norm*tt*np.exp(-1.*tt/tau)
    return SFR

def sfh_to_ssp(SFH,fage,**SFH_kwargs):

    Nage=len(fage)
    tage=np.zeros(Nage+1)
    tage[0]=0.
    for i in range(Nage-1):
        tage[i+1]=(fage[i]+fage[i+1])/2.
    tage[Nage]=2*fage[Nage-1]-tage[Nage-1]

    fssp=np.zeros(Nage)

    #pdb.set_trace()
    for i in range(Nage):
        fssp[i]=integrate.quad(lambda x: SFH(x,**SFH_kwargs),tage[i],tage[i+1])[0]

    return fssp




class model_gal():
    
    def __init__(self,SFH,velscale=70.,FWHM_gal=3.,vdisp=200.,FeH=0.,line_width=500.,EW_Ha=-99., \
            lam_range=[3600,9200],**SFH_kwargs):

        """
        Produces an model galaxy spectrum with any assumed FWHM,vdisp and EW_Ha
        Thie script relies on the bc03 template file I generated in fits format 

        :param SFH: star formation history, function, *args,**kargs
        :param velscale: desired velocity scale for the output templates library in km/s
            (e.g. 60). This is generally the same or an integer fraction of the velscale
            of the galaxy spectrum.
        :param FWHM_gal: vector or scalar of the FWHM of the instrumental resolution of
            the galaxy spectrum in Angstrom. (default 3A)
        :param vdisp: velocity dispersion of stellar components (km/s)
        :param FeH: stellar metalicity [Fe/H]
        :param line_width: emission line width (sigma of Gaussion, km/s)
        :param EW_Ha: EW of Ha, if not assigned, EW_Ha is estimated from SSFR, which is estimated from SFH
        :param lam_range: wavelength range,default [3600,9200]A
        :param **SFH_args, parameters of SFH function
        """
        #velscale=70.
        bc03 = lib.bc03(velscale,vdisp=vdisp,FWHM_gal=FWHM_gal,lam_range=lam_range)
        AM_tpl=bc03.templates

        # select metal bins
        metals=bc03.metal_grid[0,:]
        minloc=np.argmin(abs(FeH-metals))
        tpls=AM_tpl[:,:,minloc]
        fmass=bc03.fmass_ssp()[:,minloc]

        #age_bins
        ages=bc03.age_grid[:,0]
        fssp=sfh_to_ssp(SFH,ages,**SFH_kwargs)
        mass=np.dot(fmass,fssp)
        Stellar=np.dot(tpls,fssp)/mass
        #pdb.set_trace()

        wave=np.exp(bc03.log_lam_temp)
        # convert to air wavelength
        air_wave=util.vac_to_air(wave) 
        
        FWHM_line=line_width*2.36/velscale

        # emission line templates 
        MetaZ=10**(FeH)*0.02
        emission_flux, line_names, line_wave = \
                emission.emission_line_tpl(np.log(air_wave), lam_range, FWHM_line,MetaZ=MetaZ)
        
        #determin Ha EW
        Ha_flux=np.trapz(emission_flux[:,3],air_wave)
        sel=emission_flux[:,3] > 0.003
        #average continum 
        fcont=np.median(Stellar[sel])  
        #EW=10
        if  EW_Ha < 0:
            sel=np.where(ages <= 0.01)
            Nsel=len(sel[0])
            SF=np.sum(fssp[0:Nsel])
            tage=(ages[Nsel-1]+ages[Nsel])/2
            SFR=SF/tage
            SSFR=SFR/mass
            EW_Ha=63*SSFR
        
        fem=EW_Ha*fcont/Ha_flux
        Emission=fem*np.sum(emission_flux,1)
        Spectra=Stellar+Emission
    
        self.air_wave=air_wave
        self.stellar=Stellar
        self.gas_emission=Emission
        self.spectra=Spectra
        self.EW_Ha=EW_Ha
        self.stellarmass=mass 

    def model_AGN(self,Atype,fBH=0.003,Edd_ratio=1.):
        
        #BH mass: fBH*1Msolar
        #AGN bol luminosity: in unit of Lsun 
        Lbol=fBH*Edd_ratio*1.3e5/3.826
        #from Heckman 2004
        LOIII=Lbol/3500.

        # AGN templates
        if Atype == 1:
            tpl='QSO_comp.dat'
            wave,flux=readcol(tpl,comments='#',usecols=(0,1))
        elif Atype==2:
            tpl='ngc1068_template.fits'
            from astropy.io import fits
            hdu1=fits.open(tpl) 
            data1=hdu1[1].data
            wave=data1.WAVELENGTH
            flux=data1.FLUX
        else:
            raise Exception('wrong AGN type')


        #pdb.set_trace()
       
        sel=(wave > 4987) & (wave < 5027)
        fOIII=np.trapz(flux[sel],wave[sel])
        fnorm=LOIII/fOIII

        AGN_emission=fnorm*flux
        wave0=self.air_wave
        AGN_flux=np.interp(wave0,wave,AGN_emission)
        
        self.AGN_flux=AGN_flux
        self.spectra=self.gas_emission+self.stellar+AGN_flux

        

    def output_spectra(self,specname):
        from astropy.table import Table
        from astropy.table import Column
        wave=self.air_wave
        spectra=self.spectra
        stellar=self.stellar
        gas_emission=self.gas_emission
        t=Table([wave,spectra,stellar,gas_emission],names=('wave','flux','stellar','gas_emission'))
        if hasattr(self,'AGN_flux'):
            col=Column(data=self.AGN_flux,name='AGN_flux')
            t.add_column(col)

        t.write(specname,format='fits')




