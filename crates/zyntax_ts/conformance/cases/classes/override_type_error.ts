class Shape {
  scale(factor: number): number {
    return factor;
  }
}

class Label extends Shape {
  scale(factor: string): number {
    return factor.length;
  }
}
console.log(new Label().scale("ab"));
