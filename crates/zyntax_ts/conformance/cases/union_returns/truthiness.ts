function truth(x:number|boolean|string){return x?"yes":"no";}
console.log(truth(0),truth(-0),truth(NaN),truth(2),truth(""),truth("s"),truth(false),truth(true));
