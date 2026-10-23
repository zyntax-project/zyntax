const enum Choice { Unknown, Yes, No }
type YesNo = Choice.Yes | Choice.No;
function flip(c: YesNo): YesNo { return c === Choice.Yes ? Choice.No : Choice.Yes; }
let answer: YesNo = Choice.Yes;
answer = flip(answer);
const any: Choice = answer;
console.log(answer, flip(answer), any === Choice.No);
