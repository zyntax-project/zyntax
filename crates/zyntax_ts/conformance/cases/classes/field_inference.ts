class Box {
  width = 2;
  height = this.width * 3;
  big = this.height > 5;
  neg = -this.width;
  area = this.width * this.height;
}
const b = new Box();
console.log(b.width, b.height, b.big, b.neg, b.area);
