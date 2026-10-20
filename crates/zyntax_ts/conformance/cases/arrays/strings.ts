const x="hello";const a=[x,"world"];console.log(a[0]+" "+a[1]);
a[0]="new";console.log(a[0],a[1]);
function last(xs:string[]) { let result="";for(let i=0;i<xs.length;i++)result=xs[i];return result; }
console.log(last(a),last([]));
