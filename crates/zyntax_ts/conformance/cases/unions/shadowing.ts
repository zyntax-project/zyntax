function f(undefined:number):number { return undefined+1; }
console.log(f(4));
function g(n:number|undefined) { if(n!==undefined) { const n=3;console.log(n); } return n; }
console.log(g(4),g(undefined));
