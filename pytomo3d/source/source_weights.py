# calculate the weight of source based on its location and window counts
import os
import numpy as np
from pprint import pprint
from spaceweight import SpherePoint, SphereDistRel


def assign_source_to_points(sources):
    points = []
    for event, cat in sources.iteritems():
        origin = cat[0].preferred_origin()
        point = SpherePoint(origin.latitude, origin.longitude, tag=event,
                            weight=1.0)
        points.append(point)

    assert len(points) == len(sources)
    return points


def normalize_source_weights(points, wcounts):
    wsum = 0.0
    wcounts_sum = 0
    for p in points:
        wsum += p.weight * wcounts[p.tag]
        wcounts_sum += wcounts[p.tag]

    print("The summation of window counts: %d" % wcounts_sum)
    print("The iniital summation(weight * window_counts): %f" % wsum)
    factor = 1.0 / wsum

    weights = {}
    for p in points:
        weights[p.tag] = p.weight * factor

    # validate
    wsum = 0.0
    for event in weights:
        wsum += wcounts[event] * weights[event]
    if not np.isclose(wsum, 1.0):
        raise ValueError("Error normalize source weights: %f" % wsum)
    print("The normalized sum is: %f" % wsum)
    print("Final weights: %s" % weights)
    return weights


def calculate_source_weights_on_location(points, param, outputdir):
    plot_flag = param["plot"]
    # set a fake center point
    center = SpherePoint(0, 180.0, tag="Center")
    weightobj = SphereDistRel(points, center=center)

    if plot_flag:
        scan_figname = os.path.join(
            outputdir, "source_weights.smart_scan.png")
    else:
        scan_figname = None

    ref_distance, cond_number = weightobj.smart_scan(
        max_ratio=param["search_ratio"], start=0.5, gap=0.5,
        drop_ratio=0.95, plot=plot_flag,
        figname=scan_figname)

    print("Reference distance and condition number: %f, %f"
          % (ref_distance, cond_number))

    if plot_flag:
        map_figname = os.path.join(
            outputdir, "source_weights.global_map.pdf")
        weightobj.plot_global_map(figname=map_figname, lon0=180.0)

    return points


def dump_weights(weights, outputfile):
    events = weights.keys()
    events.sort()

    with open(outputfile, 'w') as fh:
        for e in events:
            fh.write("%-16s %.8e\n" % (e, weights[e]))


def calculate_source_weights(info, param, output_file, _verbose=False):
    print("=" * 10 + " Param " + "=" * 10)
    pprint(param)
    sources = {k: v["source"] for k, v in info.iteritems()}
    wcounts = {k: v["window_counts"] for k, v in info.iteritems()}

    outputdir = os.path.dirname(output_file)
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    points = assign_source_to_points(sources)
    if param["flag"]:
        print("=" * 10 + " Weight source on location " + "=" * 10)
        calculate_source_weights_on_location(
            points, param, outputdir)

    print("=" * 10 + " Normalize weights " + "=" * 10)
    weights = normalize_source_weights(points, wcounts)

    # write result out
    print("=" * 10 + " Write weights " + "=" * 10)
    print("Output weight file: %s" % output_file)
    dump_weights(weights, output_file)
