function f(c: boolean, n: number): string {
  let s: string;
  if (c) { s = "a" + "b"; } else { s = "c" + n; }
  return s + "!";
}
function g(c: boolean): number {
  let s: string;
  if (c) { s = "x" + "y"; }
  return 1;
}
function h(n: number): number {
  let s: string;
  let t = 0;
  for (let i = 0; i < n; i++) { s = "k" + i; t += 1; }
  s = "z" + "z";
  return t;
}
function r(): string {
  let s: string;
  s = "p" + "q";
  s = s + "r";
  return s;
}
console.log(f(true, 1), f(false, 2), g(true), g(false), h(3), r());
