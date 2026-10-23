function compare(text: string): boolean {
    return `v${text}` === "vother";
}
console.log(`abc${0}abc` === "abc0abc");
console.log(`n${-0}` === "n0");
console.log(`n${1e21}` === "n1e+21");
console.log(`x${`a${2}`}` === "xa2");
console.log(compare("other"));
