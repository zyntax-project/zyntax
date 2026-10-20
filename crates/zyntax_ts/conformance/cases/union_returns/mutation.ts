function mutate(n:number) {let v:number|string="seed";while(n>0){v=n;n-=1;}if(typeof v==="number")return v+1;return v;}
console.log(mutate(0),mutate(4));
function swap(x:number|string){if(typeof x==="string"){x=3;return x*2;}x="new";return x==="new"?"new!":"other!";}
console.log(swap("old"),swap(2));
