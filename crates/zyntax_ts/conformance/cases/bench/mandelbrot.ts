function mandel(size: number): number {
  let sum = 0;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const cr = 2.0 * x / size - 1.5;
      const ci = 2.0 * y / size - 1.0;
      let zr = 0, zi = 0, i = 0;
      while (i < 50 && zr * zr + zi * zi <= 4.0) {
        const t = zr * zr - zi * zi + cr;
        zi = 2.0 * zr * zi + ci;
        zr = t;
        i++;
      }
      sum += i;
    }
  }
  return sum;
}
console.log(mandel(40));
