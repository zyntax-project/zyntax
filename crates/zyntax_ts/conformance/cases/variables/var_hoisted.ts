function g(c: boolean): number {
  if (c) { var y = 1; } else { var y = 2; }
  return y + 10;
}
function loops(n: number): number {
  var i = 0;
  while (i < n) { var last = i * 2; i = i + 1; }
  return i;
}
var total = 0;
for (var k = 0; k < 3; k++) { if (k > 0) { var seen = k; total += seen; } }
console.log(g(true), g(false), loops(4), total);
