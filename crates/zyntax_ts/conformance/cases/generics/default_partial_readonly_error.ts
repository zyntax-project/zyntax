type Read={readonly x:number};
function pick<T,U=T>(x:T,y:U):U{return y;}
const p:Read={x:1};const q=pick<Read>(p,p);q.x=2;
