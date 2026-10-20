function side() { console.log(4); return 9; }
console.log(void side());
function f(n:number|undefined) { console.log(typeof n); if(typeof n==="number") return n+1; return 0; }
console.log(f(2),f(undefined));
function b(n:boolean|undefined) { if("undefined"!==typeof n) return !n; return true; }
console.log(b(false),b(undefined),typeof true,typeof "x");
const empty = ():void => undefined;
const bare = () => { return; };
console.log(empty(),bare());
