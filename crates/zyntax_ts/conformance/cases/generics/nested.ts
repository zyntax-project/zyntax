console.log(wrap(7), wrap("yes"));
function wrap<U>(x: U): U { return identity<U>(x); }
function identity<T>(x: T): T { return x; }
