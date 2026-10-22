function pickA(k: number): number {
  if (k === 0) { return 7; }
  if (k === 1) { return -7; }
  if (k === 2) { return 7; }
  if (k === 3) { return -7; }
  if (k === 4) { return 6; }
  if (k === 5) { return -6; }
  if (k === 6) { return 0; }
  if (k === 7) { return 5; }
  if (k === 8) { return 0; }
  if (k === 9) { return 5.5; }
  if (k === 10) { return -5.5; }
  if (k === 11) { return 5; }
  if (k === 12) { return 1e19; }
  if (k === 13) { return 2 ** 53 + 2; }
  if (k === 14) { return 9007199254740993; }
  if (k === 15) { return NaN; }
  if (k === 16) { return 3; }
  if (k === 17) { return Infinity; }
  if (k === 18) { return 3; }
  if (k === 19) { return -Infinity; }
  if (k === 20) { return 4; }
  if (k === 21) { return 1; }
  if (k === 22) { return 9.3e18; }
  if (k === 23) { return -9.3e18; }
  if (k === 24) { return 123456789; }
  if (k === 25) { return -0; }
  if (k === 26) { return 5; }
  if (k === 27) { return 1e300; }
  if (k === 28) { return 8; }
  if (k === 29) { return 1000000007; }
  return 0;
}
function pickB(k: number): number {
  if (k === 0) { return 3; }
  if (k === 1) { return 3; }
  if (k === 2) { return -3; }
  if (k === 3) { return -3; }
  if (k === 4) { return 3; }
  if (k === 5) { return 3; }
  if (k === 6) { return 5; }
  if (k === 7) { return 0; }
  if (k === 8) { return 0; }
  if (k === 9) { return 2; }
  if (k === 10) { return 2; }
  if (k === 11) { return 0.5; }
  if (k === 12) { return 7; }
  if (k === 13) { return 3; }
  if (k === 14) { return 10; }
  if (k === 15) { return 3; }
  if (k === 16) { return NaN; }
  if (k === 17) { return 3; }
  if (k === 18) { return Infinity; }
  if (k === 19) { return 3; }
  if (k === 20) { return 1; }
  if (k === 21) { return 4; }
  if (k === 22) { return 1000003; }
  if (k === 23) { return 1000003; }
  if (k === 24) { return 1000; }
  if (k === 25) { return 5; }
  if (k === 26) { return -0; }
  if (k === 27) { return 7; }
  if (k === 28) { return 8; }
  if (k === 29) { return 97; }
  return 0;
}
function mix(h: number, v: number): number {
  const magnitude = v !== v ? 7 : Math.abs(v) % 1000;
  const sign = v === 0 && 1 / v < 0 ? 3 : 0;
  return (h * 31 + magnitude + sign) | 0;
}
let h = 0;
for (let i = 0; i < 300000; i++) {
  const k = i % 30;
  h = mix(h, pickA(k) % pickB(k));
}
console.log(h);
for (let k = 0; k < 30; k++) {
  console.log(k, pickA(k) % pickB(k));
}
