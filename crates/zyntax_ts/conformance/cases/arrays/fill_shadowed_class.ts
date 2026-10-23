export {};
class Array {n:number;constructor(n:number){this.n=n;}fill(x:number):number[]{return [this.n+x];}}
const xs=new Array(3).fill(4);console.log(xs.length,xs[0]);
