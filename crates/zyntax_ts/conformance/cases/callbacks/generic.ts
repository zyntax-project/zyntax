function identity<T>(value: T): T { return value; }
const f = identity((n: number) => n + 0.5);
console.log(f(1.5), identity(f)(2));
