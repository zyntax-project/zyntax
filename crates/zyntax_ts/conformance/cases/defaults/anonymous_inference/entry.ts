import calculate from './library'; function score(n:number):number{let total=0;for(let i=0;i<n;i++)total+=calculate(i);return total;}console.log(score(30));
