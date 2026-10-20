function f(n: number | undefined, b: boolean) {
    if (n !== undefined) { console.log(n + 1); } else { console.log(n); }
    if (n === undefined) return 0;
    if(b) n=undefined; else n=5;
    if(n!==undefined) return n*2;
    return -1;
}
console.log(f(3,true),f(3,false),f(undefined,false));
let n: number | undefined = 1;
console.log(n+2); n=undefined; console.log(n); n=4;
const result = (n=8); console.log(result+1,n+1);
n+=2; console.log(n,n++,++n,n);
function guarded(n:number|undefined) { if(n!==undefined && n>0) return n+1; return 0; }
console.log(guarded(2),guarded(undefined));
function untouched(b:boolean) { let n:number|undefined=3; for(let i=0;i<2;i+=1) { const n=undefined; console.log(n); } return n+1; }
console.log(untouched(false));
