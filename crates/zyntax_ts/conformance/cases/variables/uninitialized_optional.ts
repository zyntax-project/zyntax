function f(c: boolean): number {
  let x: number | undefined;
  if (c) { x = 5; }
  if (x !== undefined) { return x + 1; }
  return 0;
}
let y: number | undefined;
console.log(f(true), f(false), y === undefined);
