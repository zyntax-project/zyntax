const data: number[] = [];
let squares = [0];
function fill(n: number): void { for (let i = 0; i < n; i++) data.push(i * 2); }
function sum(): number { let s = 0; for (const x of data) s += x; return s; }
function grow(): void { squares.push(squares.length * squares.length); }
function reset(): void { squares = [7, 8]; }
fill(10);
grow(); grow(); grow();
console.log(sum(), data.length, data[3], squares.length, squares[3]);
reset();
grow();
console.log(squares.length, squares[0], squares[2]);
