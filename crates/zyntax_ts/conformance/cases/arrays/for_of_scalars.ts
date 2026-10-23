function score():number {
 const ns=[1,2,3]; let sum=0;
 for(let n of ns){n+=10;sum+=n;}
 const bs=[true,false,true];for(const b of bs)if(b)sum+=100;
 const words=["a","b","c"];const copy:string[]=[];
 for(const word of words){copy.push(word);if(word==="b")sum+=1000;}
 console.log(copy[0],copy[1],copy[2]);
 const empty:number[]=[];for(const n of empty)sum+=n;
 return sum;
}
console.log(score());
