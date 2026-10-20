function f(n:number|undefined) {
 let sum=0;
 while(n!==undefined) { sum+=n; n=undefined; }
 return sum;
}
console.log(f(3),f(undefined));
function g(n:number|undefined) { for(let i=0;i<3;i+=1) { if(n!==undefined) n=n+1; } return n; }
console.log(g(2),g(undefined));
