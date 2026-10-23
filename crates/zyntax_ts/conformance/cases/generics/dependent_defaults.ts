function pick<T, U extends T = T>(x:T, y:U):U {return y;}
console.log(pick<number>(1, 3), pick("a", "b"), pick<boolean, boolean>(true, false));
