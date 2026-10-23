function grade(score: number): string {
    switch (Math.floor(score / 10)) {
        case 10:
        case 9: return "A";
        case 8: return "B";
        case 7: return "C";
        default: return "F";
    }
}
function count(n: number): number {
    let total = 0;
    switch (n) {
        case 1: total += 1;
        case 2: total += 10;
            break;
        case 3: total += 100;
        default: total += 1000;
    }
    return total;
}
console.log(grade(95), grade(83), grade(71), grade(12), count(1), count(2), count(3), count(4));
