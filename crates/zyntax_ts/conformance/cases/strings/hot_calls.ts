function greet(x: string): string { return "hello " + x; } for (let i: number = 0; i < 100; i++) { const s: string = greet("world"); if (i === 99) console.log(s); }
