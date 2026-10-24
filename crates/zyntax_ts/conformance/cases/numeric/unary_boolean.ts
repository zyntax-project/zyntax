const t = true;
const f = false;
console.log(+t, -f, -t + 1, +f);
function score(hit: boolean): number {
  return +hit * 10;
}
console.log(score(true), score(false));
