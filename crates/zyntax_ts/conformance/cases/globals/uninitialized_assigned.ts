var level: number;
let name: string;
function show(): string { return name + ":" + level * 2; }
function setup(n: number): void { level = n; name = "lvl"; }
setup(3);
console.log(show());
level++;
console.log(show(), level);
