const scale = (x: number, up: boolean): number => up ? x * 2 : x / 2;
console.log(scale(3, true), scale(3, false));
