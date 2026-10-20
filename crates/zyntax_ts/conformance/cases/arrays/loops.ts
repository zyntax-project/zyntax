function scale(xs:number[],k:number) { let sum=0; for(let i=0;i<xs.length;i++) { xs[i]*=k;sum+=xs[i]; }return sum; }
const a=[1,2,3];console.log(scale(a,2),a[0],a[1],a[2]);
function bools(xs:boolean[]) { let n=0;for(let i=0;i<xs.length;i+=1) { if(xs[i]) n+=1; }return n; }
console.log(bools([true,false,true,false,true]));
function counterValues(xs:number[]) { for(let i=0;i<xs.length;i++){ console.log(i,i+0.5); } }
counterValues([1,2,3]);
function stride(xs:number[]) { let sum=0;for(let i=0;i<xs.length;i+=2)sum+=xs[i];return sum; }
console.log(stride([1,2,3,4,5]));
