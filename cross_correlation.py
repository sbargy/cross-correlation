#!/usr/bin/env python3

# system imports
import argparse
import sys

# obspy imports
from obspy.clients.fdsn import Client
from obspy import read, UTCDateTime
from scipy import signal
from obspy.signal.cross_correlation import correlate, xcorr_max
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.core.stream import Stream

# other imports
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
                        type=int)
    parser.add_argument("-k", "--keepresponse",
                        help="don't use the remove_response call", 
                        action="store_true")
    parser.add_argument("-o", "--outfilename",
                        help="the filename for the plot output file",
                        action="store",
                        type=str)
    parser.add_argument("-v", "--verbose",
                        help="extra output for debugging",
                        action="store_true", 
                        default=False)

    args = parser.parse_args()
    # upper case the stations and channels
    args.sta = args.sta.upper()
    args.chan = args.chan.upper()

    doCorrelation(args.net, args.sta, args.chan, args.startdate, args.enddate, args.duration, \
                  args.keepresponse, args.outfilename, args.verbose)

################################################################################
def doCorrelation(net, sta, chan, start, end, duration, keep_response, outfilename, be_verbose):
    client = Client()
    stime = UTCDateTime(start)
    etime = UTCDateTime(end)
    ctime = stime
    
    mpl.rc('font',serif='Times')
    mpl.rc('font',size=16)
    
    # use same network and station
    net2 = net
    sta2 = sta
    # both 00 and 10
    loc_all = '*'
    loc1 = '00'
    loc2 = '10'
    
    # True to calculate values, False to read them from a pickle file
    # this might be desirable when debugging the plotting code piece
    calc = True
    
    print(net, net2, sta, sta2, loc_all, duration, stime, etime, keep_response)
    if calc:
        times, shifts, vals = [],[], []
        while ctime < etime:
            cnt = 1
            st = Stream()
            while cnt <= 4:
                try:
                    # get_waveforms gets 'duration' seconds of activity for the channel/date/location
                    st = client.get_waveforms(net, sta, loc_all, chan, ctime, ctime + duration, attach_response=True)
                    if not keep_response:
                        st.remove_response()
                    break
                except KeyboardInterrupt:
                    sys.exit()
                except FDSNNoDataException:
                    if be_verbose:
                        print("Exception: no data available for {} to {}".format(ctime, ctime+duration), file=sys.stderr)
                except Exception as err:
                    print(err, file=sys.stderr)
                finally:
                    cnt += 1

            if len(st) != 2:
                if be_verbose:
                    print("{} trace(s) returned for {} {} {}".format(len(st), net, sta, ctime), file=sys.stderr)
            else:
                st.filter('bandpass', freqmax=1/4., freqmin=1./8.)
                st.merge(fill_value=0)
                st.resample(1000)
                st.sort()

                try:
                    tr1 = st.select(location=loc1)[0]
                except Exception as err:
                    print(err, file=sys.stderr)
                try:
                    tr2 = st.select(location=loc2)[0]
                except Exception as err:
                    print(err, file=sys.stderr)

                # trim sample to start and end at the same times
                trace_start = max(tr1.stats.starttime, tr2.stats.starttime)
                trace_end   = min(tr1.stats.endtime, tr2.stats.endtime)
                # debug
                if be_verbose:
                    print("Before trim", file=sys.stderr)
                    print("tr1 start: {} tr2 start: {}".format(tr1.stats.starttime, tr2.stats.starttime), file=sys.stderr)
                    print("tr1 end: {} tr2 end: {}".format(tr1.stats.endtime, tr2.stats.endtime), file=sys.stderr)
                    print("max trace_start: {} min trace_end {}".format(trace_start, trace_end), file=sys.stderr)
                tr1.trim(trace_start, trace_end)
                tr2.trim(trace_start, trace_end)
                # debug
                if be_verbose:
                    print("After trim", file=sys.stderr)
                    print("tr1 start: {} tr2 start: {}".format(tr1.stats.starttime, tr2.stats.starttime), file=sys.stderr)
                    print("tr1 end: {} tr2 end: {}".format(tr1.stats.endtime, tr2.stats.endtime), file=sys.stderr)

                # calculate time offset
                time_offset = tr1.stats.starttime - tr2.stats.starttime
                cc = correlate(tr1.data, tr2.data, 500)

                # xcorr_max returns the shift and value of the maximum of the cross-correlation function
                shift, val = xcorr_max(cc)
                # append to lists for plotting
                shifts.append(shift)
                vals.append(val)
                times.append(ctime.year + ctime.julday/365.25)
    
                print("duration: {} to {} offset: {}\tshift: {} value: {}".format(ctime, ctime+duration, time_offset, shift, val))
    
            # skip 10 days for next loop
            if be_verbose:
                print("ctime: {}".format(ctime), file=sys.stderr)
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
    plt.title(net + ' ' + sta + ' ' + loc1 + ' compared to ' + net2 + ' ' + sta2 + ' ' + loc2)
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
