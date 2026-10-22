export interface Point {x:number;y:boolean} export function score(p:Point):number{return p.x+(p.y?1:0);} console.log(score({x:4,y:true}));
