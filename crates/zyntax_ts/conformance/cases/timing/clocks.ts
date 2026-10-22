const t0 = Date.now();
const p0 = performance.now();
let s = 0;
for (let i = 0; i < 2000000; i++) { s += i % 7; }
const dt = performance.now() - p0;
const wall = Date.now() - t0;
console.log(s, dt >= 0, dt < 600000, wall >= 0, wall < 600000, t0 > 1.6e12);
