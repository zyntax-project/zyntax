type Value=string|number;function echo(x:Value):number|string {return x;}
console.log(echo("x"),echo(2));let v:Value=1;v="s";console.log(v);v=3;console.log(v);
