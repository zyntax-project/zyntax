from spectral_norm import eval_AtA_times_u

u = [1] * 100
for dummy in range(10):
    v = eval_AtA_times_u(u)
    u = eval_AtA_times_u(v)
vBv = vv = 0
for ue, ve in zip(u, v):
    vBv += ue * ve
    vv += ve * ve
print(round((vBv / vv) ** 0.5, 9))
