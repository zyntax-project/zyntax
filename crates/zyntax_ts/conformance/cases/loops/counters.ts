function count(n: number): number { let c = 0; for (let i = 0; i < n; i++) { c++; } return c; }
function countLe(n: number): number { let c = 0; for (let i = 0; i <= n; i++) { c++; } return c; }
function step(n: number, k: number): number { let c = 0; for (let i = 1; i < n; i += k) { c += i; } return c; }
function shrink(n: number): number { let c = 0; for (let i = 0; i < n; i++) { n = n - 1; c += 10; } return c; }
function grow(xs: number[]): number { let c = 0; for (let i = 0; i < 6; i++) { if (xs.length < 5) { xs.push(i); } c += xs.length; } return c; }
function floats(n: number): number { let s = 0; for (let i = 1; i <= n; i++) { s += Math.sqrt(i) + i / 3 + (i % 4); } return s; }
function triangular(xs: number[]): number {
  let s = 0;
  for (let i = 0; i < xs.length; i++) { for (let j = i + 1; j < xs.length; j++) { s += xs[i] * xs[j]; } }
  return s;
}
function neighbours(xs: number[]): number {
  let s = 0;
  for (let i = 1; i < xs.length - 1; i++) { s += xs[i - 1] + xs[i] * 2 + xs[i + 1]; }
  return s;
}
function widened(xs: number[], n: number): number {
  let s = 0;
  for (let i = 0; i < n; i++) { s += xs[i]; }
  return s;
}
function weird(n: number): number { let c = 0; for (let i = 0; i < n; i++) { c++; if (c > 1000) { break; } } return c; }
const data: number[] = [3, 1, 4, 1, 5, 9, 2, 6];
console.log(count(0), count(1), count(7), count(7.5), count(-3), count(NaN), count(2 ** 20));
console.log(countLe(0), countLe(3), countLe(3.5), countLe(-1), countLe(NaN));
console.log(step(20, 3), step(5, 1), step(1, 7), step(10.5, 2));
console.log(shrink(10), shrink(1), shrink(0));
console.log(grow([1, 2]), grow([]));
console.log(floats(10), floats(0), floats(1));
console.log(triangular(data), neighbours(data), widened(data, 8), widened(data, 3), widened(data, 0));
console.log(weird(Infinity), weird(5), weird(NaN));
let total = 0;
for (let r = 0; r < 100000; r++) { total += count(r % 9) + triangular(data) + widened(data, r % 9); }
console.log(total);
