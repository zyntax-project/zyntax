function tail(n: number) { for (let i = 0; i < 1; i += 1) { if (n > 0) { return tail(n - 1); } } return 3; }
console.log(tail(5));
