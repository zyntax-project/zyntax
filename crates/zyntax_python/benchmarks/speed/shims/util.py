"""What the speed center's ``util`` gives a kernel: run its ``main`` for
the number of iterations asked and print one time per line. The
speed center's copy execs unladen_swallow's; this one is the same two
functions written out, so every interpreter runs the same code."""

import math


def run_benchmark(options, num_runs, bench_func, first=None, second=None):
    # Two extra arguments at most: scimark passes its benchmark's name
    # and its parameters, everything else nothing.
    if first is None:
        data = bench_func(num_runs)
    else:
        data = bench_func(num_runs, first, second)
    if options.take_geo_mean:
        product = 1.0
        for x in data:
            product *= x
        print(math.pow(product, 1.0 / len(data)))
    else:
        for x in data:
            print(x)


def add_standard_options_to(parser):
    parser.add_option("-n", action="store", type="int", default=100,
                      dest="num_runs", help="Number of times to run the test.")
    parser.add_option("--profile", action="store_true",
                      help="Run the benchmark through cProfile.")
    parser.add_option("--profile_sort", action="store", type="str",
                      default="time", help="Column to sort cProfile output by.")
    parser.add_option("--take_geo_mean", action="store_true",
                      help="Return the geo mean, rather than individual data.")
