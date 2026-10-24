type Direction = "north" | "south" | "east" | "west";
type Digit = 0 | 1 | 2;

function turn(d: Direction): Direction {
  if (d === "north") {
    return "east";
  }
  switch (d) {
    case "east":
      return "south";
    case "south":
      return "west";
    case "west":
      return "north";
  }
}

function score(d: Digit): number {
  return d * 10;
}

const start: Direction = "north";
let current: Direction = start;
for (let i = 0; i < 5; i++) {
  current = turn(current);
}
const one = 1;
console.log(current, score(2), score(one), turn("west"));
