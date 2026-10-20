function wide(x:number|string):number|string|boolean {return x;}
function narrow(x:number|string|boolean):number|string {if(typeof x==="boolean")return 7;return x;}
console.log(wide(2),wide("x"),narrow(false),narrow("s"),narrow(3));
