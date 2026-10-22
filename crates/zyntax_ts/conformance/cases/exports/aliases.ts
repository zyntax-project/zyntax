type Read={readonly x:number}; function keep<T>(x:T):T{return x;} export type {Read as PublicRead}; export {keep as identity}; const p:Read={x:3}; console.log(keep(p).x);
