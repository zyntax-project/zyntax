function f(n:number): number|undefined { console.log(n); if(n) return n; }
console.log(f(1)===f(2)); console.log(f(0)===f(0));
console.log(f(1),f(0),f(2));
function stop(n:number|undefined) { console.log(n!==undefined && n>0,n===undefined || n>0); }
stop(2);stop(undefined);
