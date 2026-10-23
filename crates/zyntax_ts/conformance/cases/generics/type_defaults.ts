function id<T = number>(x:T):T {return x;}
console.log(id(2), id("text"), id(true), id<number>(4));
