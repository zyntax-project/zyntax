type T = boolean;
function identity<T>(x: T) { return x; }
function choose<U>(a: U, b: U, first: boolean) { return first ? a : b; }
console.log(identity(4), identity("hello"), choose(1, 2, false), choose("a", "b", true));
