import {alias,other} from './shared';
console.log(3);
export function right(n:number):number {const xs=[n,n+1];const ys=alias(xs);ys[0]=ys[0]+1;return xs[0]+other(n);}
