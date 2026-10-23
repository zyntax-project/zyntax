const enum Choice { Unknown, Yes, No }
type YesNo = Choice.Yes | Choice.No;
let answer: YesNo = Choice.Unknown;
