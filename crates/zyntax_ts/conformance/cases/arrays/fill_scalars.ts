const zeros=new Array<number>(5).fill(0);let sum=0;for(const n of zeros)sum+=n;
const ns=Array<number>(3).fill(7);for(const n of ns)sum+=n;
const bs=new Array<boolean>(4).fill(false);let count=0;for(const b of bs)if(b)count++;
const words=new Array<string>(2).fill("hello");console.log(words[0],words[1]);
const minus=new Array<number>(2).fill(-0);console.log(1/minus[0]===-Infinity,ns.length,bs.length,sum,count);
