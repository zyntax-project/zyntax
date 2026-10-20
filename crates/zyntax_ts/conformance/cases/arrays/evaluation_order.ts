function mark(n:number) { console.log(n); return n; }
const a=[mark(1),true?mark(2):mark(3),mark(4)];console.log(a[1]);
console.log(a.push(mark(5),mark(6)),a[3],a[4]);
let i=0;const b=[10,20];b[i]+=(i=1);console.log(b[0],b[1],i);
i=0;console.log(b[i]++,b[0]);
