"""
This is a Python based barycentric correction tool.
To run the code one needs to have access to one of the ephemeride files provided by jpl:
https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/
The default JPL-eph is DE421.

Thanks to Victor, the initial version of this program is composed by V. Doreshenko.
Youli made some optimizations.
Apply barycentric correction to vector date (in MJD) assuming coordinates ra, dec given in degrees
optionally use orbit of a satellinte in heasarc format
This includes geometric and shapiro corrections, but einstein
correction is not implemented for now. Accurate to about 3e-4s.
If return_correction=True (default), correction in seconds is returned.
approx einstein is used to define number of points to calculate interpolated einstein correction
(0 means do for each data point).
see also https://asd.gsfc.nasa.gov/Craig.Markwardt/bary/ for detail and
https://pages.physics.wisc.edu/~craigm/idl/ephem.html for the codebase
which at least inspired the code below.

"""
import os
from astropy.io import fits
import numpy as np
from jplephem.spk import SPK
from tatpulsar.pulse.barycor.tdb2tdt import tdb2tdt
import tatpulsar.pulse.barycor as barydir
from scipy.interpolate import interp1d


c = 299792.458 # km/s  (kilometers per second)

def barycor(date, ra, dec,
        orbit=None, return_correction=False,
        approx_einstein=10,
        jplephem=None,
        accelerate=False):
    """
    The core function to compute the barycentric correction.

    Parameters
    ----------
    date : array-like
        The time series observed by observatory or satellite, (must in MJD)

    ra : float
        The right ascension of the observed source (in degree)

    dec : float
        The declination of the observed source (in degree)

    orbit : str, default is ``None``
       The corresponding orbit file for given time. The corresponding orbit file
       for the given time. In case of data observed by satellites,
       the position correction of the satellite orbit needs to be considered.
       If this parameter is ``None``, the geometric center of the Earth is considered as
       the position of the periodic signal detection.

       .. todo::
            the data from Earth observatory are not supported

    return_correction : boolean, default is ``False``
        Whether to returns the deviation from the pre-correction data, or the corrected time series.
        Default is ``False``, which means return the time series.

    approx_einstein : int, default is 10
        The rate of einstein correction, default is 10

    jplephem : File
        The JPL solar ephemeris file, default is DE421 stored in the packages.
        You can assign your required ephemeris, check https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/
        for details.

    accelerate : boolean, default is False
        An accelerate method to boost the computating consumption.
        The algorithm divided the time series into 60 intervals. And only one time in each
        interval are computed for barycentric correction. And interpolate the corrected time
        based on those 60 corrected time.
    """
    ra = np.radians(np.double(ra))
    dec = np.radians(np.double(dec))


    if accelerate:
        date_raw = date
        print("Accelerating barycor")
        ## Draw photons for accelerate
        N_segment = int(date_raw.size/60) # divide time into 60 segnents
        date = np.linspace(date_raw.min(), date_raw.max(), N_segment)

    jd =  np.array(date,dtype=np.float64) + 2400000.5
    if jplephem is None:
        jplephem = _get_jplfile()
    if not os.path.exists(jplephem):
        raise FileNotFoundError("File {} not found".format(jplephem))
    kernel = SPK.open(jplephem)


    msol = 0.0002959122082855911

    # get positions and velocities of the sun, earth-moon barycenter and earth
    (x_sun,y_sun,z_sun),(vx_sun,vy_sun,vz_sun) = kernel[0,10].compute_and_differentiate(jd)
    (x_em,y_em,z_em),(vx_em,vy_em,vz_em) = kernel[0,3].compute_and_differentiate(jd)
    (x_e,y_e,z_e),(vx_e,vy_e,vz_e) = kernel[3,399].compute_and_differentiate(jd)


    x_earth = x_em + x_e
    y_earth = y_em + y_e
    z_earth = z_em + z_e

    # velocities are always distance units per day as per jplephem documentation,
    # so need to divide to get km/s
    vx_earth = (vx_em + vx_e)/86400
    vy_earth = (vy_em + vy_e)/86400
    vz_earth = (vz_em + vz_e)/86400


    if orbit is not None:
        orbit = fits.open(orbit)
        mjdref = orbit[1].header['mjdreff']+orbit[1].header['mjdrefi']
        minmet = (np.min(date)-mjdref)*86400
        maxmet =(np.max(date)-mjdref)*86400

        try:
            t = orbit[1].data.field('sclk_utc')
        except:
            t = orbit[1].data.field("TIME")

        mask = (t>minmet-1)&(t<maxmet+1)
        t = t[(t>minmet-1)&(t<maxmet+1)]/86400 + mjdref

        # interpolate orbit to observed time and convert to km and km/s
        if 'pos_x' in [x.lower() for x in orbit[1].data.names]:
            x_name = 'pos_x'
            y_name = 'pos_y'
            z_name = 'pos_z'
            vx_name = 'vel_x'
            vy_name = 'vel_y'
            vz_name = 'vel_z'
        elif 'x' in [x.lower() for x in orbit[1].data.names]:
            x_name  = 'x'
            y_name  = 'y'
            z_name  = 'z'
            vx_name = 'vx'
            vy_name = 'vy'
            vz_name = 'vz'
        elif 'x_j2000' in [x.lower() for x in orbit[1].data.names]:
            x_name  = 'X_J2000'
            y_name  = 'Y_J2000'
            z_name  = 'Z_J2000'
            vx_name = 'VX_J2000'
            vy_name = 'VY_J2000'
            vz_name = 'VZ_J2000'
        else:
            raise IOError("The position and the velocity columns are not found")

        x_s = np.interp(date, t, orbit[1].data.field(x_name)[mask]/1000.)
        y_s = np.interp(date, t, orbit[1].data.field(y_name)[mask]/1000.)
        z_s = np.interp(date, t, orbit[1].data.field(z_name)[mask]/1000.)

        vx_s = np.interp(date, t, orbit[1].data.field(vx_name)[mask]/1000.)
        vy_s = np.interp(date, t, orbit[1].data.field(vy_name)[mask]/1000.)
        vz_s = np.interp(date, t, orbit[1].data.field(vz_name)[mask]/1000.)



        x_obs, y_obs, z_obs = x_earth + x_s, y_earth + y_s, z_earth + z_s
        vx_obs, vy_obs, vz_obs = vx_earth + vx_s, vy_earth + vy_s, vz_earth + vz_s

        # orbital correction
        ocor = (vx_earth*x_s+vy_earth*y_s+vz_earth*z_s)/c**2
    else:
        x_obs, y_obs, z_obs = x_earth, y_earth , z_earth
        vx_obs, vy_obs, vz_obs = vx_earth , vy_earth , vz_earth
        ocor = 0.

    # #components of the object unit vector:
    x_obj = np.cos(dec)*np.cos(ra)
    y_obj = np.cos(dec)*np.sin(ra)
    z_obj = np.sin(dec)

    #geometric correction
    geo_corr = (x_obs*x_obj + y_obs*y_obj + z_obs*z_obj)/c

    #einstein correction
    if approx_einstein == 0:
        einstein_corr = tdb2tdt(jd)
    else:
        xx = np.linspace(jd.min(),jd.max(),approx_einstein)
        einstein_corr = tdb2tdt(xx)
        einstein_corr = np.interp(jd,xx,einstein_corr)
    #shapiro correction ("Shapiro") = - (2 G Msun/c^3) log(1 + cos th)
    sun_dist = np.sqrt((x_sun-x_obs)**2+(y_sun-y_obs)**2+(z_sun-z_obs)**2)
    costh = ((x_obs-x_sun)*x_obj+(y_obs-y_sun)*y_obj + (z_obs-z_sun)*z_obj)/sun_dist
    shapiro_corr = - 9.8509819e-06*np.log(1.+costh)
    corr = geo_corr + ocor + einstein_corr - shapiro_corr

    ## Interpolate for accelerated photons
    if accelerate:
        ## interpolate
        corr_fun = interp1d(date, corr, kind='quadratic')
        corr = corr_fun(date_raw)
        if return_correction:
            return corr
        else:
            return date_raw + corr/86400.

    if return_correction:
        return corr
    else:
        return date + corr/86400.

def _get_jplfile(jpleph='de421'):
    """
    return the absolute path of JPL solar-ephemeris file in the barycor package
    ``jpleph`` set the vesion of ephemeris file, default is de421, and the "de421.bsp"
    file will return.
    """
    return os.path.join(barydir.__path__[0], jpleph+'.bsp')
