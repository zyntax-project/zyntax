function invoke(f: (n: number) => void, n: number): void { f(n); }
invoke(n => { console.log(n); }, 2);
const discard: (n: number) => void = n => n + 1;
discard(4);
const empty = () => {};
empty();
const result = (n: number): number => n + 1;
const ignored = result as (n: number) => void;
ignored(2);
console.log(7);
