function first(xs:number[]):number {for(const x of xs)return x;return -1;}
console.log(first([8,9]),first([]));
function firstRow(rows:number[][]):number[] {for(const row of rows)return row;return [];}
function escaped(n:number):number {const row=firstRow([[n],[7]]);row.push(2);return row[0]+row[1];}
let sum=0;for(let i=0;i<200;i++)sum+=escaped(i);console.log(sum);
