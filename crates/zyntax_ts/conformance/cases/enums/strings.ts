enum Color { Red = "red", Green = "green", Blue = `blue` }
function label(c: Color): string { return "color:" + c; }
let c: Color = Color.Green;
console.log(label(c), label(Color.Blue), c === Color.Green, Color.Red === "red");
const s: string = Color.Red;
console.log(s === "red" ? s : "none");
