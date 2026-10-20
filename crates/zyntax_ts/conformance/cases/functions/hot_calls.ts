function step(x: number): number { return x * 2 + 1; } let sum: number = 0; for (let i: number = 0; i < 1000; i++) { sum += step(i); } console.log(sum);
