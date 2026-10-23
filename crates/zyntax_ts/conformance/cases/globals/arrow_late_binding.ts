let x = 1;
const f = () => x;
x = 2;
console.log(f());
const g = () => y;
const y = 3;
console.log(g());
