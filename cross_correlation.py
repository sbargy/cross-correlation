#!/usr/bin/env python3

# system imports
import argparse
import sys

# obspy imports
from obspy.clients.fdsn import Client
from obspy import read, read_inventory, UTCDateTime
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
    parser.add_argument("-i", "--interval",
                        help="interval in minutes to skip between segments",
                        action="store",
                        default="14400",
                        type=int)
    parser.add_argument("-k", "--keepresponse",
                        help="don't use the remove_response call", 
                        action="store_true")
    parser.add_argument("-o", "--outfilename",
                        help="the filename for the plot output file",
                        action="store",
                        type=str)
    parser.add_argument("-r", "--responsefilepath",
                        help="the path to the response file location, the filename is generated in code",
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
                  args.interval, args.keepresponse, args.outfilename, args.responsefilepath, args.verbose)

################################################################################
def doCorrelation(net, sta, chan, start, end, duration, interval, 
                  keep_response, outfilename, resp_filepath, be_verbose):
    stime = UTCDateTime(start)
    etime = UTCDateTime(end)
    ctime = stime
    skiptime = 24*60*60*10 # 10 days in seconds, TODO make a command line parameter
    skiptime = interval*60 #

    # location constants
    LOC00 = '00'
    LOC10 = '10'
    
    # True to calculate values, False to read them from a pickle file
    # this might be desirable when debugging the plotting code piece
    calc = True
    
    print(net, sta, LOC00, LOC10, duration, interval, stime, etime, keep_response, resp_filepath)
    if calc:
        times, shifts, vals = [],[], []
        while ctime < etime:
            cnt = 1
            attach_response = True

            if resp_filepath:
                inv00 = read_inventory(f'{resp_filepath}/RESP.{net}.{sta}.{LOC00}.{chan}', 'RESP')
                inv10 = read_inventory(f'{resp_filepath}/RESP.{net}.{sta}.{LOC10}.{chan}', 'RESP')
                attach_response = False

            st00 = getStream(net, sta, LOC00, chan, ctime, duration, be_verbose, attach_response)
            st10 = getStream(net, sta, LOC10, chan, ctime, duration, be_verbose, attach_response)

            if len(st00) == 0:
                if be_verbose:
                    print("no traces returned for {} {} {} {} {}".format(net, sta, LOC00, chan, ctime), file=sys.stderr)
                ctime += skiptime
                continue

            if len(st10) == 0:
                if be_verbose:
                    print("no traces returned for {} {} {} {} {}".format(net, sta, LOC10, chan, ctime), file=sys.stderr)
                ctime += skiptime
                continue

            if len(st00) > 1:
                if be_verbose:
                    print("gap(s) found in segment for {} {} {} {} {}".format(net, sta, LOC00, chan, ctime), file=sys.stderr)
                ctime += skiptime
                continue

            if len(st10) > 1:
                if be_verbose:
                    print("gap(s) found in segment for {} {} {} {} {}".format(net, sta, LOC10, chan, ctime), file=sys.stderr)
                ctime += skiptime
                continue

            if ((st00[0].stats.endtime - st00[0].stats.starttime) < (duration - 1.0/st00[0].stats.sampling_rate)):
                if be_verbose:
                    print("skipping short segment in {} {} {} {} {}".format(net, sta, LOC00, chan, ctime), file=sys.stderr)
                ctime += skiptime
                continue

            if ((st10[0].stats.endtime - st10[0].stats.starttime) < (duration - 1.0/st10[0].stats.sampling_rate)):
                if be_verbose:
                    print("skipping short segment in {} {} {} {} {}".format(net, sta, LOC10, chan, ctime), file=sys.stderr)
                ctime += skiptime
                continue

            if not attach_response:
                st00.attach_response(inv00)
                st10.attach_response(inv10)

            if not keep_response:
                st00.remove_response()
                st10.remove_response()

            # apply a bandpass filter and merge before resampling
            st00.filter('bandpass', freqmax=1/4., freqmin=1./8., zerophase=True)
            st00.merge(fill_value=0)
            st00.resample(1000)

            st10.filter('bandpass', freqmax=1/4., freqmin=1./8., zerophase=True)
            st10.merge(fill_value=0)
            st10.resample(1000)

            # get the traces from the stream for each location
            try:
                tr1 = st00.select(location=LOC00)[0]
            except Exception as err:
                print(err, file=sys.stderr)
            try:
                tr2 = st10.select(location=LOC10)[0]
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
            ctime += skiptime
    
        # persist the data in a pickle file
        if outfilename:
            with open(outfilename + '.pickle', 'wb') as f:
                pickle.dump([shifts, vals, times], f)
        else:
            with open(net + '_' + sta + '_' + net + '_' + sta + '.pickle', 'wb') as f:
                pickle.dump([shifts, vals, times], f)
    else:
        # retrieve the data from the pickle file
        if outfilename:
            with open(outfilename + '.pickle', 'rb') as f:
                shifts, vals, times = pickle.load(f) 
        else:
            with open(net + '_' + sta + '_' + net + '_' + sta + '.pickle', 'rb') as f:
                shifts, vals, times = pickle.load(f) 
    
    
    mpl.rc('font',serif='Times')
    mpl.rc('font',size=16)
    
    fig = plt.figure(1, figsize=(10,10))
    
    plt.subplot(2,1,1)
    plt.title(net + ' ' + sta + ' ' + LOC00 + ' compared to ' + net + ' ' + sta + ' ' + LOC10)
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
        plt.savefig(net + '_' + sta + '_' + net + '_' + sta + '.PDF', format='PDF')

################################################################################
def getStream(net, sta, loc, chan, ctime, duration, be_verbose, attach_response):
    cnt = 1
    client = Client()
    st = Stream()

    while cnt <= 4:
        try:
            # get_waveforms gets 'duration' seconds of activity for the channel/date/location
            # only attach response if we're not using a response file
            if attach_response:
                st = client.get_waveforms(net, sta, loc, chan, ctime, ctime + duration, attach_response=True)
            else:
                st = client.get_waveforms(net, sta, loc, chan, ctime, ctime + duration)
            break
        except KeyboardInterrupt:
            sys.exit()
        except FDSNNoDataException:
            if be_verbose:
                print(f"No data available for {net}.{sta}.{loc}.{chan} {ctime} to {ctime+duration}", file=sys.stderr)
        except Exception as err:
            print(err, file=sys.stderr)
        finally:
            cnt += 1

    return st

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
