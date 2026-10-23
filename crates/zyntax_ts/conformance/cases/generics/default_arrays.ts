function size<T = number>():number {return 0;}
function keep<T = number>(xs:T[]):T[] {return xs;}
const xs=[4];const ys=keep(xs);ys.push(8);
console.log(size(), size<boolean>(), xs[0], xs[1], ys.length);
