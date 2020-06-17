#!/usr/bin/env python3

# system imports
import argparse
import sys

from obspy.clients.fdsn import Client
from obspy import read, UTCDateTime
from scipy import signal
from obspy.signal.cross_correlation import correlate, xcorr_max

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle

################################################################################
def main():
    parser = argparse.ArgumentParser(description="Cross correlate sensor streams", formatter_class=SmartFormatter)
    parser.add_argument("net",
                        help="Network code (e.g. II)",
                        action="store")
    parser.add_argument("sta",
                        help="Station Code (e.g. MSEY or WRAB)",
                        action="store")
    parser.add_argument("chan",
                        help="channel (e.g. BHZ or BH0",
                        action="store")
    parser.add_argument("startdate",
                        help="R|start date (YYYY-JJJ OR\n"
                                           "YYYY-MM-DD), UTC is assumed",
                        action="store")
    parser.add_argument("enddate",
                        help="R|end date (YYYY-JJJ OR\n"
                                         "YYYY-MM-DD), UTC is assumed",
                        action="store")
    parser.add_argument("-d", "--duration",
                        help="the duration in seconds of the sample",
                        action="store",
                        type = int)
    parser.add_argument("-k", "--keepresponse",
                        help="don't use the remove_response call", 
                        action="store_true")
    parser.add_argument("-o", "--outfilename",
                        help="the filename for the plot output file",
                        action="store",
                        type = str)

    args = parser.parse_args()
    # upper case the stations and channels
    args.sta = args.sta.upper()
    args.chan = args.chan.upper()

    doCorrelation(args.net, args.sta, args.chan, args.startdate, args.enddate, args.duration, \
                  args.keepresponse, args.outfilename)

################################################################################
def doCorrelation(net, sta, chan, start, end, duration, keepresponse, outfilename):
    client = Client()
    stime = UTCDateTime(start)
    etime = UTCDateTime(end)
    ctime = stime
    
    mpl.rc('font',serif='Times')
    mpl.rc('font',size=16)
    
    net2= net
    sta2 = sta
    loc = '00'
    loc2 = '10'
    
    # True to calculate values, False to read them from a pickle file
    # this might be desirable when debugging the plotting code piece
    calc = True
    
    print(net, net2, sta, sta2, loc, loc2, duration, ctime, ctime+duration, keepresponse)
    if calc:
        times, shifts, vals = [],[], []
        while ctime < etime:
            print("duration: ", ctime, " to ", ctime + duration)
            try:
                # get_waveforms gets 'duration' seconds of activity for the channel/date/location
                st = client.get_waveforms(net, sta, loc, chan, ctime, ctime + duration, attach_response=True)
                st += client.get_waveforms(net2, sta2, loc2, chan, ctime, ctime + duration, attach_response=True)
    
                # parameters for remove_response
                #
                # output (str)
                # Output units, one of:
                #    "DISP" displacement, output unit is meters
                #    "VEL"  velocity, output unit is meters/second <---default value
                #    "ACC"  acceleration, output unit is meters/second**2
                #
                # water_level (float) Water level for deconvolution.
                #
                # pre_filt
                # Apply a bandpass filter in frequency domain to the data before deconvolution. 
                # The list or tuple defines the four corner frequencies (f1, f2, f3, f4) of a 
                # cosine taper which is one between f2 and f3 and tapers to zero 
                # for f1 < f < f2 and f3 < f < f4.

                if not keepresponse:
                    st.remove_response() # is it correct to remove response before resampling?
    
                st.resample(1000)
                st.sort()
    
                # the following call returns a function to perform cross-correlation of two signals 
                # up to a specified maximal shift. A value of 100 shifts 50 samples in either direction
                cc_func = correlate(st[0].data, st[1].data, 100)
    
                # xcorr_max returns the shift and value of the maximum of the cross-correlation function
                shift, val = xcorr_max(cc_func)
                if shift > 40.:
                    shift = 40.
                if shift < -40:
                    shift = -40.
                # append to lists for plotting
                shifts.append(shift)
                vals.append(val)
                times.append(ctime.year + ctime.julday/365.25)
    
                print('\tshift: ', shift, ' value: ', val)
    
            except:
                print('\twaveforms not available')
                pass
    
            # skip 10 days for next loop
            ctime += 24*60*60*10
    
        # persist the data in a pickle file
        if outfilename:
            with open(outfilename + '.pickle', 'wb') as f:
                pickle.dump([shifts, vals, times], f)
        else:
            with open(net + '_' + sta + '_' + net2 + '_' + sta2 + '.pickle', 'wb') as f:
                pickle.dump([shifts, vals, times], f)
    else:
        # retrieve the data from the pickle file
        if outfilename:
            with open(outfilename + '.pickle', 'rb') as f:
                shifts, vals, times = pickle.load(f) 
        else:
            with open(net + '_' + sta + '_' + net2 + '_' + sta2 + '.pickle', 'rb') as f:
                shifts, vals, times = pickle.load(f) 
    
    
    fig = plt.figure(1, figsize=(10,10))
    
    plt.subplot(2,1,1)
    plt.title(net + ' ' + sta + ' ' + loc + ' compared to ' + net2 + ' ' + sta2 + ' ' + loc2)
    plt.plot(times, shifts,'.')
    plt.ylabel('Time Shift (ms)')
    
    plt.subplot(2,1,2)
    plt.plot(times, vals, '.')
    #plt.ylim((0.8, 1.0))
    plt.ylim((0, 1.0))
    plt.xlabel('Time (year)')
    plt.ylabel('Correlation')
    
    if outfilename:
        plt.savefig(outfilename + '.PDF', format='PDF')
    else:
        plt.savefig(net + '_' + sta + '_' + net2 + '_' + sta2 + '.PDF', format='PDF')

################################################################################
class SmartFormatter(argparse.HelpFormatter):
    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)

################################################################################
if __name__ == '__main__':
    main()
