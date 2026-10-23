function value():number{console.log("unexpected fill");return 1;}
console.log("before");
const xs=new Array<number>(Infinity).fill(value());
console.log(xs.length);
