type Read={readonly x:number};
function pick<T,U=Read>(x:T,y:U):U{return y;}
const p:Read={x:1};const q=pick<number>(0,p);q.x=2;
