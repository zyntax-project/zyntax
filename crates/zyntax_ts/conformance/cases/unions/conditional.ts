function f(b:boolean) { return b ? 2 : undefined; }
function g(n:number|undefined) { return n===undefined ? 9 : n+1; }
console.log(f(true),f(false),g(3),g(undefined));
function nested(a:boolean,b:boolean) { return a ? (b ? 1 : undefined) : 2; }
console.log(nested(true,true),nested(true,false),nested(false,true));
function make(n:number):number|undefined { if(n>0)return n; }
function guardedNested(n:number):number|undefined { const m=make(n);return m!==undefined?(m>1?m:undefined):undefined; }
console.log(guardedNested(3),guardedNested(1),guardedNested(0));
