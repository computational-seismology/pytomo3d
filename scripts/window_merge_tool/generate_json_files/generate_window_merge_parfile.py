import os
import json
import glob
import argparse

superbase = "/lustre/atlas/proj-shared/geo111/Wenjie/DATA_SI/ASDF"

windowbase = os.path.join(superbase, "window")
outputbase = os.path.join(windowbase, "mt_input")
period_list = ["50_100", "60_100"]
#period_list = ["50_100", ]

if not os.path.exists(parfile_dir):
    os.makedirs(parfile_dir)


def read_txt_into_list(txtfile):
    with open(txtfile, 'r') as f:
        content = f.readlines()
        eventlist = [ line.rstrip() for line in content]
    return eventlist


def generate_json_dirfiles(eventlist):
    content = []

    for event in eventlist:
        print "="*20
        print "Event:", event
        for period in period_list:
            parlist = {}
            parlist['input_file'] = \
                os.path.join(windowbase, "%s.%s" % (event, period),
                             "windows.json")
            parlist['output_file'] = \
                os.path.join(outputbase, "%s.%s.json" % (event, period))
            content.append(parlist)

    par_jsonfile = "window_merge.dir.json"
    print("outputfile: %s" % par_jsonfile)
    with open(par_jsonfile, 'w') as f:
        json.dump(content, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', action='store', dest='eventlist_file', required=True)
    args = parser.parse_args()

    eventlist = read_txt_into_list(args.eventlist_file)
    generate_json_dirfiles(eventlist)
