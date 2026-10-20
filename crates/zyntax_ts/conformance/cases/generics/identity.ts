function identity<T>(x: T): T { return x; }
console.log(identity(3), identity<boolean>(false), identity<string>("ok"));
