function id<T>(n:T):T { return n; }
function f(n:number|undefined) { return id(n); }
console.log(f(2),f(undefined),id< number | undefined >(3));
