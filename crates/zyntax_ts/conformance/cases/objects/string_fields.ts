interface Named { name: string; size: number }
type Tagged = { tag: string; on: boolean };
class Person {
    name: string = "anon";
    title = "none";
    age: number;
    constructor(age: number) { this.age = age; }
    rename(): void { this.name = "renamed"; }
    label(): string { return this.title + ":" + this.name; }
}
function describe(n: Named): string { return n.name + "/" + n.size; }
const item: Named = { name: "box", size: 3 };
const t: Tagged = { tag: `t${1}`, on: true };
const p = new Person(30);
console.log(describe(item), t.tag, t.on, p.label());
p.rename();
item.name = "crate";
const names: string[] = [item.name, p.name];
console.log(p.label(), describe(item), names.length, names[1]);
