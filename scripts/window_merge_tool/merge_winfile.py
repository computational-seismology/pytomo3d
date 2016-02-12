import os
import glob
from util import read_json_file, JSONObject, isclose
import argparse
import json


class WindowMerge(object):
    """
    Rewrite output from pyflex into a simpler form(not all
    information are necessary in next stage) and
    also do some processing to the window(add weighting)
    """

    def __init__(self, dirfile, verbose):
        self.dirfile = dirfile
        self._verbose = verbose

    def smart_run(self, strategy="combined"):
        """
        running window merging

        :param strategy: strategy to merge windows. values could be 
            "combined" and "selective". 
            If "combined", each channel will be kept and the weighting will be
            calculated according to the number of windows. 
            If "selective", only one channel(the one with max number of window)
            will be kept and the weighting will be 1. If there are multiple max
            values, the channel_id with smaller location value will be kept.
        :type strategy: string
        """
        strategy = strategy.lower()
        if strategy not in ['combined', 'selective']:
            raise ValueError("strategy can only be: 1)'combined'; "
                             "2)'selective'")

        dirlist = read_json_file(self.dirfile)
        if isinstance(dirlist, list):
            for _dir in dirlist:
                self.merge_window(_dir, strategy=strategy)
        elif isinstance(dirlist, JSONObject):
            self.merge_window(dirlist, strategy)

    def merge_window(self, dir_obj, strategy):
        input_winfile = dir_obj.input_file
        output_winfile = dir_obj.output_file

        if self._verbose:
            print("+"*20)
            print("input window file: %s" % input_winfile)
            print("output window file: %s" % output_winfile)
            print("merging strategy: %s" % strategy)

        windows = self.load_input_winfile(input_winfile)
        new_windows = {}
        for sta_name, sta_win in windows.iteritems():
            if self._verbose == 2:
                print("="*15+"\nStation: %s" % sta_name)
            new_sta_win = self.merge_one_station(sta_win, strategy)
            new_windows[sta_name] = new_sta_win

        if self._verbose == 1:
            num_sta, num_win = self._stats_window(windows)
            print("Before merging, number of station and window: [%d, %d]" %
                  (num_sta, num_win))
            num_sta, num_win = self._stats_window(new_windows)
            print("Before merging, number of station and window: [%d, %d]" %
                  (num_sta, num_win))

        self.write_output(new_windows, output_winfile)

    @staticmethod
    def _stats_window(windows):
        num_sta = 0
        num_win = 0
        for sta_name, sta_win in windows.iteritems():
            num_sta += 1
            for channel_id, channel_win in sta_win.iteritems():
                num_win += len(channel_win)
        return num_sta, num_win
        
    @staticmethod
    def find_duplicate_channel(window, channel_id):

        def _split_channel_id(channel_id):
            """
            Channel id: network.station.location.channel
            """
            return channel_id.split(".")

        numwin = {}
        info = _split_channel_id(channel_id)

        for _id, _win in window.iteritems():
            newinfo = _split_channel_id(_id)
            if newinfo[-1][-1:] == info[-1][-1:]:
                numwin[_id] = len(_win)
        return numwin
            
    @staticmethod
    def calculate_weighting(channel_id, winnum, strategy):

        def combined_weighting(channel_id, winnum):
            return float(winnum[channel_id]) / sum(winnum.values())

        def selective_weighting(channel_id, winnum):
            values = winnum.values()
            values.sort()
            if winnum[channel_id] == max(values):
                if values[-1] == values[-2]:
                    # multiple max, select one
                    _channel_list = []
                    for _id, _num in winnum.iteritems():
                        if _num == max(values):
                            _channel_list.append(_id)
                    if channel_id == min(_channel_list):
                        weighting = 1.0
                    else:
                        weighting = 0.0
                else:
                    weighting = 1.0
            else:
                weighting = 0.0

            return weighting
        
        if strategy == "combined":
            return combined_weighting(channel_id, winnum)
        elif strategy == "selective":
            return selective_weighting(channel_id, winnum) 
        else:
            raise NotImplementedError("strategy not implemented:%s" 
                                      % strategy)

    def merge_one_station(self, window, strategy):
        new_window = {}
        for channel_id, channel_win in window.iteritems():
            winnum = self.find_duplicate_channel(window, channel_id)
            if len(winnum) == 1:
                weighting = 1.0
            if len(winnum) > 1:
                weighting = self.calculate_weighting(
                                channel_id, winnum, strategy)

            if self._verbose == 2:
                print("%s" % str(channel_id)),
                print(winnum), 
                print(" --> weighting: %.2f" % (weighting))

            if isclose(weighting, 0.0):
                continue

            new_window[channel_id] = []
            for win in channel_win:
                newwin = {}
                newwin["initial_weighting"] = weighting
                channel_id = win["channel_id"]
                content = channel_id.split(".")
                newwin["obsd_id"] = channel_id
                newwin["synt_id"] = "%s.%s.S3.MX%s" % (content[0], content[1], 
                                             content[3][-1])
                newwin["relative_starttime"] = win["relative_starttime"]
                newwin["relative_endtime"] = win["relative_endtime"]
                new_window[channel_id].append(newwin)
        return new_window

    @staticmethod
    def float_array_to_string(array):
        return ['{:.3f}'.format(i) for i in array] 

    @staticmethod
    def load_input_winfile(input_winfile):
        with open(input_winfile) as fh:
            return json.load(fh)

    @staticmethod
    def write_output(windows, output_winfile):
        dirname = os.path.dirname(output_winfile)
        if not os.path.exists(dirname):
            os.makedirs(dirname) 
        if os.path.exists(output_winfile):
            print("Output_winfile exists and removed: %s" % output_winfile)
            os.remove(output_winfile)

        with open(output_winfile, 'w') as fh:
            json.dump(windows, fh, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', action='store', dest='files', required=True)
    parser.add_argument('-v', "--verbosity", action="count", 
                        dest='verbose', help="increase output verbosity")
    args = parser.parse_args()

    if args.verbose is None:
        args.verbose = 0
    winmer = WindowMerge(args.files, args.verbose)
    #winmer.smart_run(strategy="selective")
    winmer.smart_run(strategy="combined")

