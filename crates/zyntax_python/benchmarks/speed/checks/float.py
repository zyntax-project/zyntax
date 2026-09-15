import float as kernel
p = kernel.benchmark(1000)
print(p.x, p.y, p.z)
q = kernel.benchmark(100000)
print(q.x, q.y, q.z)
