import scimark
rnd = scimark.Random(7)
x = rnd.RandomVector(32)
scimark.FFT_transform(32, x)
print([round(v, 9) for v in list(x)[:6]])
scimark.FFT_inverse(32, x)
print([round(v, 9) for v in list(x)[:6]], scimark.FFT_num_flops(16))
print(scimark.FFT(["16", "2"]))
