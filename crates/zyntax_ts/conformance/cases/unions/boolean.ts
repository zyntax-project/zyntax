function make(b: boolean): boolean | undefined { if (b) return false; }
function choose(n: boolean | undefined) { if (n !== undefined) return !n; return true; }
console.log(make(true), make(false), choose(false), choose(undefined));
function truth(n: boolean | undefined) { if(n) return 1; return 2; }
console.log(truth(true), truth(false), truth(undefined));
