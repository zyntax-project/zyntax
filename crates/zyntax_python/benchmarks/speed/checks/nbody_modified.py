import nbody_modified as nb
nb.offset_momentum(nb.BODIES["sun"])
print(nb.report_energy())
nb.advance(0.01, 20000)
print(nb.report_energy())
for name in ["sun", "jupiter", "saturn", "uranus", "neptune"]:
    r, v, m = nb.BODIES[name]
    print(name, r, v, m)
