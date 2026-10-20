function print(x: number) { console.log(x); }
function bare(x: number) {
    if (x < 0) return;
    print(x);
    return;
}
function forward(x: number) { return print(x); }
print(1);
bare(-1);
bare(2);
forward(3);
