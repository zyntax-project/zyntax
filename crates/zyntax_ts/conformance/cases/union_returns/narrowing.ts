function read(x:number|boolean|string) {
 if(typeof x==="number")return x*2;
 if(typeof x==="boolean")return x?10:-10;
 return x==="s"?"s!":"other!";
}
console.log(read(3),read(false),read("s"));
function negative(x:number|string) { if("string"!==typeof x)return x+1;return x==="x"?"x?":"other?"; }
console.log(negative(4),negative("x"));
