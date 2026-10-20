function identity<T>(x: T): T { return x; }
function step(n: number): number { return identity<number>(n) + 0.5; }
let result = 0; for (let i = 0; i < 500; i++) result = step(result);
console.log(result);
