function f(a:number|undefined,b:number|undefined) { console.log(a===b,a!==b,a==b,a!=b,a===3,a===undefined); }
f(3,3);f(3,4);f(undefined,3);f(3,undefined);f(undefined,undefined);f(NaN,NaN);
function empty():void { console.log(7); }
console.log(undefined===empty(),empty()!==undefined);
function compareVoid(n:number|undefined) { console.log(n===empty()); }
compareVoid(undefined);compareVoid(3);
