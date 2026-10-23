class A {private x:number;constructor(n:number){this.x=n;}}
class B {private x:number;constructor(n:number){this.x=n;}}
function keep<T extends A>(x:T):T{return x;}
keep(new B(1));
