const BASE = 10;
enum Size { Small = BASE, Medium = Small * 2, Large = Size.Medium + BASE, Huge = `${"x"}`.length + 100 }
console.log(Size.Small, Size.Medium, Size.Large, Size["Large"]);
