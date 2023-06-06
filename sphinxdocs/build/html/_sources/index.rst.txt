.. TAT-pulsar documentation master file, created by
   sphinx-quickstart on Jun 24 00:00:00 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TAT-pulsar documentation
========================

A very useful toolbox for your high energy pulsar timing analysis.


.. tabs::

    .. tab:: Installation


        Download the whole repository to your local directory using ``git clone`` or ``git fetch``.

        For example, In you local path execute:

        .. code-block:: console

           git clone https://github.com/tuoyl/TAT-pulsar.git

        And you will get the folder ``tat-pulsar``, in the folder you will see a file named ``setup.py``. We will install the whole package based on this script.

        The most elegant thing to do before installing is to create a new conda environment to avoid conflicts with your existing python environment.

        .. code-block:: console

            conda create -n tatpulsar-env python=3

        after downloading the dependancies, execute

        .. code-block:: console

            conda activate tatpulsar-env

        to enter the tatpulsar-env environment of conda, you will see (``tatpulsar-env``) before the shell prompt.


        Now you are all set to install the repository. In the directory where the setup.py located, execute:

        .. code-block:: console

            python3 -m pip install -e .


        if you want to uninstall the package for generating the product.

        .. code-block:: console

            python3 -m pip uninstall tatpulsar


    .. tab:: Navigation








Contents
--------

.. toctree::
   :maxdepth: 2

    Tutorials <tutorials.rst>
    TAT-pulsar API <autodocs.rst>
