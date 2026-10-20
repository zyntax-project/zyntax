function choose(b:boolean,n:number) {return b?n:"x";}
console.log(choose(true,2),choose(false,3));
function nested(a:boolean,b:boolean) {return a?1:b?false:"s";}
console.log(nested(true,false),nested(false,true),nested(false,false));
