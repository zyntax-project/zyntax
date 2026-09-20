import chaos
chaos.main(2)
with open("py.ppm", "rb") as f:
    data = f.read()
print(len(data), data[:15], sum(data) % 100003)
