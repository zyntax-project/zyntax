class A { n: number = 1; get(): number { return this.n; } }
function f(a: A): number { return a.get(); }
console.log(f({ n: 1 }));
