function f(n:number|undefined) { return n ?? 9; }
function b(n:boolean|undefined) { return n ?? true; }
console.log(f(0),f(undefined),f(4),b(false),b(undefined));
function side(n:number):number { console.log(n); return n; }
function g(n:number|undefined) { return n ?? side(7); }
console.log(g(0),g(undefined));
