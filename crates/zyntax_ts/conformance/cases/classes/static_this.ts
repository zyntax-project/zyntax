class C {
  static a = 1;
  static b = this.a + 1;
  static twice(): number {
    return this.b * 2;
  }
  static total(): number {
    return this.twice() + C.a;
  }
}
class D extends C {
  static c = 2;
  static d = this.c + 1;
  static e = super.a + this.c + 1;
  static {
    D.c = D.c + 100;
  }
}
console.log(C.a, C.b, D.c, D.d, D.e, C.total(), D.total());
