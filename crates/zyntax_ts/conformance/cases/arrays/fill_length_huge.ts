function value():number{console.log("unexpected fill");return 1;}
console.log("before");
const xs=new Array<number>(1e300).fill(value());
console.log(xs.length);
