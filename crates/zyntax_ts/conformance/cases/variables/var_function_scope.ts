var x = 1;
var x = x + 2;
function f(n: number): number {
  var total = 0;
  for (var i = 0; i < n; i++) { var sq = i * i; total += sq; }
  return total;
}
console.log(x, f(4));
