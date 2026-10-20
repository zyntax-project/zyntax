function get(xs:number[]) { if(xs.length<=1)return -1;return xs[1]; }
function get2(xs:number[]) { if(2<xs.length)return xs[2];return 0; }
console.log(get([]),get([1,2]),get2([1,2,3]));
const a=[1,2];let i=1;console.log(a[i]);i=0;console.log(a[i]);
function early(xs:number[]) { if(xs.length===0)return -1;return xs[0]; }
function truth(xs:number[]) { if(!xs.length)return -1;return xs[0]; }
console.log(early([]),early([3]),truth([]),truth([4]));
