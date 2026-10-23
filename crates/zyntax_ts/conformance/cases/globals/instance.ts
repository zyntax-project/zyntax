class Stats { sum: number = 0; count: number = 0; }
let stats = new Stats();
function record(x: number): void { stats.sum += x; stats.count++; }
function mean(): number { return stats.sum / stats.count; }
function restart(): void { stats = new Stats(); }
record(2); record(4); record(9);
console.log(mean(), stats.count);
restart();
record(10);
console.log(mean(), stats.count);
