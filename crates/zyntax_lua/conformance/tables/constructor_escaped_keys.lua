-- bracketed string keys decode their escapes like any string literal
local t = {['\n'] = 'N', ['\65'] = 'dec', ['\x42'] = 'hex', ['\u{43}'] = 'utf8', ['a\tb'] = 'tab', ["\""] = 'quote', ['\\'] = 'bs', ['\z
    D'] = 'z'}
print(t[string.char(10)], t.A, t.B, t.C, t['a' .. string.char(9) .. 'b'], t['"'], t[string.char(92)], t.D)
local esc = {['&'] = '&amp;', ['<'] = '&lt;', ['\n'] = '\\n', ['\t'] = '\\t'}
print((("a<b\n&c\t"):gsub('[&<\n\t]', esc)))
local n = 0
for k in pairs(t) do n = n + 1; if #k ~= 1 and k ~= 'a\tb' then print('odd key', k) end end
print(n)
