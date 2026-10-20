function make() { return (n: number) => twice(n); }
const f = twice;
function twice(n: number) { return n * 2; }
console.log(f(3), make()(4));
