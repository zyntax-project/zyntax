class A { constructor(public readonly n: number) {} bump(): A { return new A(this.n + 1); } }
const a = new A(1).bump().bump();
console.log(a.n);
interface HasN { n: number }
function g(h: HasN): number { return h.n; }
console.log(g(a));
