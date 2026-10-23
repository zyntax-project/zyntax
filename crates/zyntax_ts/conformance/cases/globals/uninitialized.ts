let total: number;
let ready: boolean;
function start(): void { total = 10; ready = true; }
function add(x: number): number { total += x; return total; }
start();
add(5);
console.log(total, ready, add(1));
