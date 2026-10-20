function first<T>(xs:T[],fallback:T):T{if(xs.length>0)return xs[0];return fallback;} first<number>([1],"x");
