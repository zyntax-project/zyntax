function pick(c: boolean): number {
  let r: number;
  if (c) { r = 1; } else { r = 2; }
  return r * 10;
}
let total: number;
total = 5;
let flag: boolean;
flag = total > 3;
var count: number;
count = 0;
for (let i = 0; i < 3; i++) { count += i; }
console.log(pick(true), pick(false), total, flag, count);
