from astropy.io import fits   

hdu1=fits.open('seyfert1_template.fits') 
data1=hdu1[1].data
wave1=data1.WAVELENGTH
flux1=data1.FLUX

hdu2=fits.open('seyfert2_template.fits') 
data2=hdu2[1].data
wave2=data2.WAVELENGTH
flux2=data2.FLUX


