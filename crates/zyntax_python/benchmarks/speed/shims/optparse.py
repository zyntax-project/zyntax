"""The part of ``optparse`` the kernels use, for an interpreter without
the module: the options the speed center's ``util`` declares and the
``--benchmark`` scimark adds, ``-n <int>``, ``--name=value``,
``--flag``, and the positional arguments left over. CPython and PyPy
run with their own ``optparse``; this file is placed beside a kernel
only for zypy. Written without ``*args``, ``setattr`` or dynamic
attributes: the destinations are the fields of ``Values``."""

import sys


class Values:
    def __init__(self):
        self.num_runs = 100
        self.profile = False
        self.profile_sort = "time"
        self.take_geo_mean = False
        self.benchmark = None

    def set(self, dest, value):
        if dest == "num_runs":
            self.num_runs = value
        elif dest == "profile":
            self.profile = value
        elif dest == "profile_sort":
            self.profile_sort = value
        elif dest == "take_geo_mean":
            self.take_geo_mean = value
        elif dest == "benchmark":
            self.benchmark = value
        else:
            raise ValueError("option destination " + dest + " is not one this shim holds")


class Option:
    def __init__(self, name, action, kind, default, dest):
        self.name = name
        self.action = action
        self.kind = kind
        self.default = default
        self.dest = dest


class OptionParser:
    def __init__(self, usage="", description=""):
        self.usage = usage
        self.description = description
        self.options = []

    def add_option(self, name, action="store", type=None, default=None, dest=None, help=None):
        if dest is None:
            dest = name
            while dest.startswith("-"):
                dest = dest[1:]
            dest = dest.replace("-", "_")
        self.options.append(Option(name, action, type, default, dest))

    def parse_args(self, argv=None):
        if argv is None:
            argv = sys.argv[1:]
        values = Values()
        for option in self.options:
            values.set(option.dest, option.default)
        rest = []
        i = 0
        while i < len(argv):
            arg = argv[i]
            i += 1
            if not arg.startswith("-"):
                rest.append(arg)
                continue
            text = None
            if "=" in arg:
                cut = arg.index("=")
                text = arg[cut + 1:]
                arg = arg[:cut]
            found = False
            for option in self.options:
                if option.name != arg:
                    continue
                found = True
                if option.action == "store_true":
                    values.set(option.dest, True)
                elif option.action == "store_false":
                    values.set(option.dest, False)
                else:
                    if text is None:
                        text = argv[i]
                        i += 1
                    if option.kind == "int":
                        values.set(option.dest, int(text))
                    elif option.kind == "float":
                        values.set(option.dest, float(text))
                    else:
                        values.set(option.dest, text)
            if not found:
                raise ValueError("unknown option " + arg)
        return values, rest
