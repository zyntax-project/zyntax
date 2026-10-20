function f(n:number) { if(n===0) return undefined; if(n===1) return 2; return f(n-1); }
function g(n:number):number|undefined { if(n===0) return; return g(n-1); }
console.log(f(0),f(1),f(4),g(3));
