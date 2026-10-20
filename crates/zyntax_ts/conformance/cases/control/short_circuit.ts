function hit(x: number): boolean { console.log(x); return true; } console.log(false && hit(1), true || hit(2), true && hit(3));
