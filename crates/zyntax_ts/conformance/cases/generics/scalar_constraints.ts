function add<T extends number = number>(x:T,y:T):number {return x+y;}
function square<T extends number>(x:T):number {return x*x;}
function label<T extends string = string>(x:T):string {return x+"!";}
function forward<U extends number>(x:U):number {return add<U>(x,x);}
function choose<T extends number>(flag:boolean,x:T):number {return flag?x:0;}
function same<T extends number>(flag:boolean,x:T,y:T):T {return flag?x:y;}
console.log(add(2,4), square(3), label("yes"), forward(5), choose(true,8), choose(false,8), same(true,1,2));
