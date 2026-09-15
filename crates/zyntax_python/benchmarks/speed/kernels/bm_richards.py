#! /usr/bin/env python2.4

"""perf.py wrapper for the classic Richards benchmark.

See richards.py for copyright and history information.
"""

__author__ = "fijall@gmail.com (Maciej Fijalkowski)"
__contact__ = "collinwinter@google.com (Collin Winter)"


# Python imports
import optparse
import time
import sys
if sys.version_info[0] > 2:
    xrange = range

# Local imports
import richards
import util


def test_richards(iterations):
    # Warm-up
    r = richards.Richards()
    r.run(iterations=2)

    times = []
    for _ in xrange(iterations):
        t0 = time.time()
        r.run(iterations=1)
        t1 = time.time()
        times.append(t1 - t0)
    return times

if __name__ == "__main__":
    parser = optparse.OptionParser(
        usage="%prog [options]",
        description="Test the performance of the Richards benchmark")
    util.add_standard_options_to(parser)
    options, args = parser.parse_args()

    util.run_benchmark(options, options.num_runs, test_richards)
