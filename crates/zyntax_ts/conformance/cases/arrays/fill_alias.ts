class Item {value:number;constructor(n:number){this.value=n;}bump():number{this.value+=1;return this.value;}}
function score():number {
 const row=[1];const grid=new Array<number[]>(3).fill(row);
 grid[0].push(2);grid[1][0]=5;
 const item=new Item(10);const items=new Array<Item>(3).fill(item);
 let total=0;for(const x of items)total+=x.bump();
 return total+grid[2][0]+grid[2].length+item.value;
}
let total=0;for(let i=0;i<200;i++)total+=score();console.log(total);
