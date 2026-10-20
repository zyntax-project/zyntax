function mark(n:number){console.log(n);if(n>0)return n;return "cold";}
console.log(mark(1)===mark(-1),mark(-2)!==mark(-3));
console.log(mark(-4),mark(5),typeof mark(-6));
