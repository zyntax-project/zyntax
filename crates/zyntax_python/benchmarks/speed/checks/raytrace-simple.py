import raytrace_simple
raytrace_simple._main()
with open("test_raytrace.ppm", "rb") as f:
    data = f.read()
print(len(data), data[:12], sum(data) % 100003)
