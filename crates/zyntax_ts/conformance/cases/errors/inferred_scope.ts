function caller(x: number) { return callee(); }
function callee() { return x; }
console.log(caller(3));
