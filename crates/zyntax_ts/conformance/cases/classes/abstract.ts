abstract class Shape {
  constructor(public scale: number) {}
  abstract area(): number;
  describe(): number {
    return this.area() * this.scale;
  }
}
class Square extends Shape {
  constructor(public side: number) {
    super(2);
  }
  area(): number {
    return this.side * this.side;
  }
}
class Circle extends Shape {
  radius = 1;
  area(): number {
    return 3 * this.radius * this.radius;
  }
}
function total(shapes: Shape[]): number {
  let sum = 0;
  for (const s of shapes) {
    sum = sum + s.describe() + s.area();
  }
  return sum;
}
console.log(total([new Square(3), new Circle(5)]));
