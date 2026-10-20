function message(n: number) { if (n <= 0) return "done"; return message(n - 1); }
console.log(message(5));
