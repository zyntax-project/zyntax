function pick<T,U extends T>(x:T,y:U):U{return y;} pick<number,string>(1,"bad");
