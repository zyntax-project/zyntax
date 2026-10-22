import keep from './library';const a=[2,4,8];const b=keep(a);b[1]+=3;console.log(a[1]);const s=keep(['a','b']);console.log(s[1]);const flags=keep([true,false]);console.log(flags[1]);
