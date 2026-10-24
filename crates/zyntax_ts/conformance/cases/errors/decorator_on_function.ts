declare function dec(t: any): any;

@dec
function f(): number {
  return 1;
}
console.log(f());
