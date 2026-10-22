export default function factorial(n:number):number{if(n<=1)return 1;return n*factorial(n-1);} export {factorial}; console.log(factorial(2));
