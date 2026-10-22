console.log(1);
export function keep<T>(v:T):T{return v;}
export function make(n:number):number[]{return [n,n+1];}
export function alias(xs:number[]):number[]{return xs;}
export function other(n:number):number{return n+3;}
