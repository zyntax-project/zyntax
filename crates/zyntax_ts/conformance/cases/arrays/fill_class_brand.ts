class A {private x:number;constructor(n:number){this.x=n;}}
class B {private x:number;constructor(n:number){this.x=n;}}
const xs=new Array<A>(2).fill(new B(3));
