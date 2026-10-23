function keep<T extends number[]>(x:T):T {return x;}
function first<T extends number>(xs:T[]):T {return xs[0];}
const xs=[3,4];const ys=keep(xs);ys[1]=8;console.log(first(xs),xs[1]);
