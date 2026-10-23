let i = 0;
let sum = 0;
do {
    i++;
    if (i % 2 === 0) continue;
    sum += i;
} while (i < 9);
let once = 0;
do { once++; } while (false);
let x: number;
do { x = 5; } while (once > 10);
console.log(i, sum, once, x);
