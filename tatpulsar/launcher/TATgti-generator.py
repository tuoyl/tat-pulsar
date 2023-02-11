#!/usr/bin/env python
"""
The GUI tool to generate GTI file or a list of GTI from the input of the user
selected by GUI interaction
"""

import argparse
import numpy as np
from astropy.io import fits

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
try:
    from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk as PlotNav
except:
    from matplotlib.backends.backend_tkagg import NavigationToolbar2TkAgg as PlotNav
from matplotlib.figure import Figure
import matplotlib.pylab as plt
from matplotlib import gridspec
from matplotlib.widgets import SpanSelector, Button, TextBox
plt.style.use(['nature', 'science'])

# check python version to import correct version of tkinter
import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
    from Tkinter.filedialog import asksaveasfile
else:
    import tkinter as Tk
    from tkinter.filedialog import asksaveasfile

class App:
    def __init__(self, root, args):
        self.args = args
        self.gtis = []

        # Generating lightcurves from event files
        self.generate_lightcurve()

        # create a container for buttons
        buttonFrame = Tk.Frame(root)
        buttonFrame.pack()

        # create buttons
        self.buttonSavefile = Tk.Button(master=buttonFrame,
                                text='save',
                                command=self.save_file)
        self.buttonSavefile.pack(side=Tk.LEFT, pady=10)
        if self.args.savegti:
            self.buttonSavefile['state'] = "disable"
        self.buttonQuit = Tk.Button(master=buttonFrame,
        			           text='Quit',
        			           command=root.destroy)
        self.buttonQuit.pack(side=Tk.LEFT)

        # create container for text
        textFrame = Tk.Frame(root)
        textFrame.pack()

        # create text
        self.label = Tk.Label(master=textFrame,
        					  text=self.present_info(),
        					  justify=Tk.LEFT)
        self.label.pack()

        # create container for plot
        plotFrame = Tk.Frame(root)
        plotFrame.pack(side=Tk.BOTTOM)

        # create plot
        self.fig = Figure(figsize=(5, 4), dpi=250)
        self.ax = self.fig.add_subplot(111)
        #self.ax.set_xlim([-0.2, 1.2])
        #self.ax.set_ylim([-0.2, 1.2])
        self.plot_lightcurve()

        self.canvas = FigureCanvasTkAgg(self.fig, master=plotFrame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()
        PlotNav(self.canvas, root)
        self.GUI_time_selection()

    def present_info(self):
        text = \
                f"""
                Number of files: {len(self.args.eventfile)}\n
                Total Time Elapse: {self.data.max() - self.data.max()}\n
                Selected GTI exposure: 0
                """
        return text

    def update_info(self, exposure):
        text = \
                f"""
                Number of files: {len(self.args.eventfile)}\n
                Total Time Elapse: {self.data.max() - self.data.max()}\n
                Selected GTI exposure: {exposure}
                """
        self.label.config(text=text)


    def generate_lightcurve(self):
        """
        read events files and generate lightcurves
        """
        self.data = np.array([])
        for file in self.args.eventfile:
            print(f'File {file} is loaded')
            hdulist = fits.open(file)
            self.data = np.append(self.data,
                             hdulist[self.args.extnum].data[self.args.colname])
        assert self.data.size != 0, "Datasets are empty!"
        y, x = np.histogram(self.data,
                            bins=np.arange(self.data.min(),
                                           self.data.max(),
                                           self.args.binsize))
        self.time = x[:-1]
        self.counts = y


    def plot_lightcurve(self):
        """
        """
        t0 = self.time.min()
        mask = (self.counts>0)
        self.time = self.time[mask]
        self.counts = self.counts[mask]

        self.ax.errorbar(self.time-t0,
                         self.counts,
                         np.sqrt(self.counts),
                         color='black', fmt='.-')
        self.ax.set_xlabel(f"Time - {t0} (s)")
        self.ax.set_ylabel("Counts")


    def GUI_time_selection(self):
        """
        select the GTI manually and return the list
        """
        def toggle_selector(event):
            #TODO press key to interact
            pass

        def onselect(xmin, xmax):
            t0 = self.time.min()
            self.gtis.append([xmin + t0,
                              xmax + t0])
            self.ax.axvspan(xmin, xmax, facecolor='gray', alpha=0.3)
            self.fig.canvas.draw()
            self._print_gti()
            self.update_info(xmax-xmin)

        self.span = SpanSelector(self.ax, onselect, 'horizontal', useblit=True,
                props=dict(alpha=0.5, facecolor='red'), interactive=True)

    def _print_gti(self):
        if self.gtis == []:
            print("GTI is empty!")
        else:
            print("updated GTI list is:")
            for gti in self.gtis:
                print(f'{gti[0]}, {gti[1]}')

    def save_file(self):
        f = asksaveasfile(initialfile = 'Untitled.txt',
                          defaultextension=".txt",
                          filetypes=[("All Files","*.*"),("Text Documents","*.txt")],
                          mode='w')
        for gti in self.gtis:
            f.write(f'{gti[0]} {gti[1]}\n')
        print("File saved successfully")

def parse_args():
    #required
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-e","--eventfile", nargs='+', help="The list of event file", required=True)
    parser.add_argument("-b","--binsize", type=int, help="The binsize (in sec) of the lightcurve", required=True)
    parser.add_argument("-c", "--colname", type=str,
                        help="The name of Column to be loaded, default is 'Time'", required=False, default='TDB')
    parser.add_argument("-x", "--extnum", type=str,
                        help="The extension number to load event data (index start from 0)", required=False, default=1)

    #optional
    parser.add_argument("--dpi", type=int, help='dpi value for output figure, default is 250', required=False,
                        default=250)
    parser.add_argument("--savegti", type=str, help="save the GTI lists to a TXT file", required=False)

    return parser.parse_args()

def main():

    args = parse_args()

    root = Tk.Tk()
    root.title("Correlation Examples")
    app = App(root, args)
    root.mainloop()

if __name__ == "__main__":
    main()
