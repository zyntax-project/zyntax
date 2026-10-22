// The dividend is invariant in the loop, so a conversion of it must not run before its range test.
function sweep(x: number): number {
  let last = 0;
  for (let i = 0; i < 300000; i++) { last = x % ((i % 7) + 1); }
  return last;
}
console.log(sweep(1e19));
console.log(sweep(NaN));
console.log(sweep(Infinity));
console.log(sweep(-Infinity));
console.log(sweep(-12345));
console.log(sweep(9.3e18));
console.log(sweep(2 ** 53 + 2));
console.log(sweep(1e300));
console.log(sweep(0));
console.log(sweep(123456789));
