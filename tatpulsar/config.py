"""Functions related to TATpulsar configuration."""

import os
import pkg_resources

__all__ = ["jpleph"]

def jpleph(ephemeris):
    """Full path to the requested JPL Solar ephemeris file

    Parameters
    ----------
    ephemeris : str
        available names are {'de200', 'de421'}

    Returns
    -------
    str
        Full path to the requested file

    Notes
    -----
    This is **not** for files needed at runtime. Those are located by :func:`pint.config.runtimefile`.  This is for files needed for the example notebooks.
    """
    filename = ephemeris.lower() + '.bsp'
    return pkg_resources.resource_filename(
        __name__, os.path.join("pulse/barycor/", filename)
    )
