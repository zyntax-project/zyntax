function make(n:number):number[] { return [n,n+1,n+2]; }
function alias(xs:number[]):number[] { return xs; }
const a=make(3);const b=alias(a);if(b.length>1)b[1]=8;
console.log(a[0],a[1],a[2],b.length);
function words():string[] { return ["a","b","c"]; }
const w=words();if(w.length>1)console.log(w[1]);
