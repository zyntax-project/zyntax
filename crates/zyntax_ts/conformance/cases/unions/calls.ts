type Maybe = number | undefined;
function make(b: boolean): Maybe { if (b) return 3; return undefined; }
function echo(n: Maybe): Maybe { return n; }
function read(n: Maybe): number { if (n === undefined) return -1; return n + 1; }
console.log(make(true), make(false), echo(4), echo(undefined));
console.log(read(make(true)), read(echo(undefined)));
function branches(b:boolean):number|undefined { return b ? 5 : 6; }
console.log(branches(true),branches(false));
