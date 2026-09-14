# Mandelbrot: 875 x 500 points, up to 1000 iterations each, summed
# through a palette. Floating-point arithmetic and a data-dependent
# inner loop; the classic escape-time kernel.
# Returns 112789639.

def palette_sum(iteration: int, max_iter: int) -> int:
    fraction = iteration / max_iter
    r = int(fraction * 255.0)
    g = int((1.0 - fraction) * 255.0)
    b = int((0.5 - abs(fraction - 0.5)) * 2.0 * 255.0)
    return r + g + b

def main() -> int:
    size = 25
    max_iter = 1000
    max_rad = 65536.0
    width = 35 * size
    height = 20 * size
    scale = 0.004
    checksum = 0
    for y in range(height):
        for x in range(width):
            cx = x * scale - 2.5
            cy = y * scale - 1.0
            zx = 0.0
            zy = 0.0
            zx2 = 0.0
            zy2 = 0.0
            iter = 0
            while iter < max_iter:
                if zx2 + zy2 >= max_rad:
                    break
                new_zx = zx2 - zy2 + cx
                new_zy = 2.0 * zx * zy + cy
                zx = new_zx
                zy = new_zy
                zx2 = zx * zx
                zy2 = zy * zy
                iter += 1
            checksum += palette_sum(iter, max_iter)
    return checksum

print(main())
