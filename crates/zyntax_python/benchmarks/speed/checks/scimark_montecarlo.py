import scimark
print(scimark.MonteCarlo_integrate(20000))
rnd = scimark.Random(113)
print([round(rnd.nextDouble(), 12) for i in range(5)])
print(scimark.MonteCarlo(["1000"]))
