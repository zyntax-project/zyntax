class Item {value:number;constructor(n:number){this.value=n;}bump():number{this.value+=1;return this.value;}}
function rows():number[][] {return [[1],[2]];}
function score():number {
 const saved:number[][]=[];for(const row of rows())saved.push(row);
 saved[0].push(9);let sum=0;for(const row of saved)for(const n of row)sum+=n;
 const items=[new Item(10),new Item(20)];for(const item of items)sum+=item.bump();
 return sum+saved[0].length+items[0].value;
}
let total=0;for(let i=0;i<200;i++)total+=score();console.log(total);
