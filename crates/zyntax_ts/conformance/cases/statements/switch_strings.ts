function code(name: string): number {
    let result = -1;
    switch (name) {
        case "zero": result = 0; break;
        case "one": result = 1; break;
        case "two": { result = 2; break; }
    }
    return result;
}
console.log(code("one"), code("two"), code("zero"), code("three"));
