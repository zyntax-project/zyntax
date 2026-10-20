function choose(flag:boolean) { if(flag)return 1;return "x"; }
console.log(choose(true),choose(false),typeof choose(false));
function all(n:number) { if(n>0)return n;if(n===0)return false;return "cold"; }
console.log(all(2),all(0),all(-1));
