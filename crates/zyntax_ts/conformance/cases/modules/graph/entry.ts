import {left} from './left';
import {right} from './right';
import './shared';
console.log(left(5)+right(5));
let total=0; for(let i=0;i<100;i++) total=total+left(i)+right(i); console.log(total);
