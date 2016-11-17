from __future__ import (absolute_import, division, print_function)
import json


def load_json(filename):
    with open(filename) as fh:
        return json.load(fh)


def dump_json(content, filename):
    with open(filename, 'w') as fh:
        json.dump(content, fh, indent=2, sort_keys=True)
