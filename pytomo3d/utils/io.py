from __future__ import (absolute_import, division, print_function)
import json


def load_json(filename):
    with open(filename) as fh:
        return json.load(fh)


def dump_json(content, filename):
    with open(filename, 'w') as fh:
        json.dump(content, fh, indent=2, sort_keys=True)


def check_dict_keys(dict_to_check, keys):
    if not isinstance(dict_to_check, dict):
        raise TypeError("input dict_to_check should be type of dict: %s"
                        % (type(dict_to_check)))

    set_input = set(dict_to_check.keys())
    set_stand = set(keys)

    if set_input != set_stand:
        print("More: %s" % (set_input - set_stand))
        print("Missing: %s" % (set_stand - set_input))
        raise ValueError("Keys is not consistent: %s --- %s"
                         % (set_input, set_stand))
