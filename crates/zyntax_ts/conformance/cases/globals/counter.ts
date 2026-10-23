let count = 0;
let step = 1;
function tick(): void { count += step; }
function reset(): void { count = 0; step = step * 2; }
for (let i = 0; i < 5; i++) tick();
console.log(count);
reset();
tick(); tick();
console.log(count, step);
