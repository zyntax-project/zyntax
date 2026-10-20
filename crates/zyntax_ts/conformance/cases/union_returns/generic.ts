function id<T>(x:T):T {return x;}
function choose(n:number) {return n>0?n:"cold";}
function f(n:number) {return id(choose(n));}
console.log(f(3),f(-1),f(0));
