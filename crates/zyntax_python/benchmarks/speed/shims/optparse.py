"""The part of ``optparse`` the kernels use, for an interpreter without
the module: options with a default, ``-n <int>``, ``--name=value`` and
``--flag``, and the positional arguments left over. CPython and PyPy
run with their own ``optparse``; this file is placed beside a kernel
only for zypy."""

import sys


class Values:
    pass


class OptionParser:
    def __init__(self, usage="", description=""):
        self.usage = usage
        self.description = description
        self.options = []

    def add_option(self, *names, **settings):
        self.options.append((names, settings))

    def parse_args(self, argv=None):
        if argv is None:
            argv = sys.argv[1:]
        values = Values()
        by_name = {}
        for names, settings in self.options:
            dest = settings.get("dest")
            if dest is None:
                dest = names[-1].lstrip("-").replace("-", "_")
            setattr(values, dest, settings.get("default"))
            for name in names:
                by_name[name] = (dest, settings)
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
                arg, text = arg.split("=", 1)
            if arg not in by_name:
                raise SystemExit("unknown option " + arg)
            dest, settings = by_name[arg]
            action = settings.get("action", "store")
            if action == "store_true":
                setattr(values, dest, True)
            elif action == "store_false":
                setattr(values, dest, False)
            else:
                if text is None:
                    text = argv[i]
                    i += 1
                if settings.get("type") == "int":
                    setattr(values, dest, int(text))
                elif settings.get("type") == "float":
                    setattr(values, dest, float(text))
                else:
                    setattr(values, dest, text)
        return values, rest
