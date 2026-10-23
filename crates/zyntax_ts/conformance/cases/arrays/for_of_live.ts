function pick(xs:number[],log:number[]):number[]{log.push(7);return xs;}
function score():number {
 let xs=[1,2];const original=xs;const log:number[]=[];let sum=0;
 for(const n of pick(xs,log)){sum+=n;xs=[99];if(n===1)original.push(3);}
 console.log(log.length,xs[0],original.length);return sum;
}
console.log(score());
