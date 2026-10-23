function makeLength(log:number[]):number{log.push(1);return 3;}
function value(log:number[]):number{log.push(2);return 9;}
const log:number[]=[];const ns=new Array<number>(makeLength(log)).fill(value(log));
const empty=new Array<number>(-0).fill(value(log));
console.log(log[0],log[1],log[2],log.length,ns[2],empty.length);
