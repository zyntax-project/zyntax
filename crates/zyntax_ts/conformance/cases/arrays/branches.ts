function f(flag:boolean) { let a=[1,2];if(flag)a=[4];else a=[8,9];if(a.length>0)return a[0];return 0; }
console.log(f(true),f(false));
const a=[1,2];const b=a;if(a.length>1)console.log(b[1]);
