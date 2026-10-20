function choose(n:number):number|string|boolean{if(n>0)return n;if(n===0)return false;return "cold";}
function compare(a:number|string|boolean,b:number|string|boolean){return a===b;}
console.log(compare(choose(1),choose(1)),compare(choose(-1),"cold"),compare(choose(0),false),compare(choose(1),"1"));
console.log(compare(NaN,NaN),compare(-0,0),compare("a","a"),compare(false,0));
