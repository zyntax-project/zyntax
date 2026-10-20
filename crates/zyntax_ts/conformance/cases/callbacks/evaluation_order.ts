type Transform = (n: number) => number;
function make(): Transform { console.log(1); return n => n * 2; }
function argument(): number { console.log(2); return 3; }
console.log(make()(argument()));
