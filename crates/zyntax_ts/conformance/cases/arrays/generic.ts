function first<T>(xs:T[],fallback:T):T { if(xs.length>0) return xs[0];return fallback; }
function through<U>(xs:Array<U>,fallback:U):U { return first<U>(xs,fallback); }
console.log(first([1.5,2],0),first([3],0),first<number>([],9));
console.log(through(["hello","world"],"missing"),through<string>([],"empty"),first([false,true],true));
function identity<T>(x:T):T { return x; }
const a=[5,6];const b=identity(a);b[1]=9;console.log(a[1]);
const word=identity("static");const words=[word];console.log(words[0]);
