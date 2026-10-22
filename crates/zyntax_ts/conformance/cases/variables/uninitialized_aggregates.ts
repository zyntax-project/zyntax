type P = { x: number; y: number };
function a(c: boolean): number {
  let xs: number[];
  if (c) { xs = [1, 2, 3]; } else { xs = [4, 5]; }
  let t = 0;
  for (let i = 0; i < xs.length; i++) { t += xs[i]; }
  return t + xs.length;
}
function b(c: boolean): number {
  let p: P;
  if (c) { p = { x: 1, y: 2 }; } else { p = { x: 3, y: 4 }; }
  p.x = p.x * 10;
  return p.x + p.y;
}
function d(c: boolean): number {
  let xs: number[];
  if (c) { xs = [1]; }
  return 7;
}
function e(n: number): number {
  let t = 0;
  let p: P;
  for (let i = 0; i < n; i++) { p = { x: i, y: i }; t += p.x; }
  return t;
}
console.log(a(true), a(false), b(true), b(false), d(true), d(false), e(4));
