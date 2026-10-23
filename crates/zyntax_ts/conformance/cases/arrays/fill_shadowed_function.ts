export {};
class Holder {n:number;constructor(n:number){this.n=n;}fill(x:number):number[]{return [this.n+x];}}
function Array(n:number):Holder{return new Holder(n);}
const xs=Array(3).fill(4);console.log(xs.length,xs[0]);
