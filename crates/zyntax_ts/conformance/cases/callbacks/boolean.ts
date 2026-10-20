function apply(f: (b: boolean) => boolean, b: boolean): boolean { return f(b); }
let f = (b: boolean) => !b;
console.log(apply(f, false), f(true));
