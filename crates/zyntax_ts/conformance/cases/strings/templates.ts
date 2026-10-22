const n = 3;
const s = `n=${n} half=${n / 2} ok=${n > 2} neg=${-0} big=${1e21}`;
console.log(s);
console.log("a" + 1 + true, 1 + "b");
console.log(`${"x"}${"y"}`, `plain`);
