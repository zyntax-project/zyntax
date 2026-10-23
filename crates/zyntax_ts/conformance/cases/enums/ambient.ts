declare const enum Level { Low = 1, High = Low * 10 }
declare enum Remote { A, B }
let level: Level = Level.High;
console.log(level, Level.Low);
