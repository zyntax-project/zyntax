function peek(): number {
  return Config.size;
}
console.log("start");
console.log(peek());
class Config {
  static size = 4;
}
