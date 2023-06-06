.. TAT-pulsar documentation master file, created by
   sphinx-quickstart on Jun 24 00:00:00 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TAT-pulsar documentation
========================

Hey there! This is a Python toolbox made for folks interested in messing around with pulsar timing analysis, especially in high-energy scenarios. You know those pesky corrections needed for stuff like the barycentric correction and binary systems? Yeah, we've got functions for those. Plus, we've included some timing analysis tools and some examples to help you get started with your pulsar timing exploration. Enjoy!

.. tabs::

    .. tab:: pip Installation


        1. Open your terminal or command prompt.

        2. Type in the following command:

        .. code-block:: console

            pip install tat-pulsar

        3. Let pip handle the rest. You should see messages indicating that TAT-Pulsar was successfully installed.


    .. tab:: source code Installation

        1. Open your terminal or command prompt.

        2. Navigate to the directory where you want to clone the GitHub repository using the 'cd' command:


        .. code-block:: console

            cd /path/to/your/directory

        3.  Clone the TAT-Pulsar repository with the following command:

        .. code-block:: console

            git clone https://github.com/tuoyl/TAT-pulsar.git

        Once you follow these steps, you'll find a new folder named 'tat-pulsar'. Inside this folder, there's a file called 'setup.py' - this little script is our ticket to installing the entire package.But before we dive into installation, here's a pro tip: it's a smart move to create a new conda environment first. Why? Well, it helps keep things clean and tidy, avoiding any clashes with your existing Python environment.

        4. create a conda Environment

        .. code-block:: console

            conda create -n tatpulsar-env python=3
            conda activate tatpulsar-env

        5. install the source code

        .. code-block:: console

            python3 -m pip install -e .


    .. tab:: Navigation

Contents
--------

.. toctree::
   :maxdepth: 2

    Tutorials <tutorials.rst>
    TAT-pulsar API <autodocs.rst>
