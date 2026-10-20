type Transform = (n: number) => number;
function relay(f: Transform): Transform { return f; }
function factory(): () => Transform { return () => n => n + 0.25; }
const f = factory()();
const higher = (g: Transform, n: number) => g(n);
console.log(relay(f)(1), higher(n => n * 2, 3));
