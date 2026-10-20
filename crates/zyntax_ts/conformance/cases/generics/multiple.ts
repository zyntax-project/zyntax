function second<A, B>(a: A, b: B): B { return b; }
console.log(second(1, "two"), second<string, boolean>("x", true));
