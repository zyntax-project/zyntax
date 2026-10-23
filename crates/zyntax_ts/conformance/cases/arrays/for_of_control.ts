function sum():number {
 let total=0; const x=99;
 for(const x of [1,2,3,4,5]){if(x===2)continue;if(x===5)break;total+=x;}
 for(const row of [[1,2],[3,4]])for(const n of row){if(n===3)continue;total+=n;}
 return total+x;
}
console.log(sum());
