-- a table stored in a field of its own constructor's shape after nil:
-- linked lists, binary trees and doubly linked nodes
local function new(l) return {left = l} end
local leaf = new(nil)
local node = new(leaf)
print(node.left == leaf, leaf.left, node.left.left)

-- singly linked list built by prepending
local function cons(v, rest) return {value = v, next = rest} end
local list = nil
for i = 1, 5 do list = cons(i, list) end
local sum, n = 0, 0
local p = list
while p do sum = sum + p.value; n = n + 1; p = p.next end
print(sum, n, list.value, list.next.value)

-- binary search tree
local function tnode(k) return {key = k, left = nil, right = nil} end
local function insert(t, k)
  if t == nil then return tnode(k) end
  if k < t.key then t.left = insert(t.left, k) else t.right = insert(t.right, k) end
  return t
end
local root
for _, k in ipairs({50, 30, 70, 20, 40, 60, 80, 35}) do root = insert(root, k) end
local out = {}
local function walk(t)
  if t then walk(t.left); out[#out + 1] = t.key; walk(t.right) end
end
walk(root)
print(table.concat(out, " "))
local function depth(t) if not t then return 0 end return 1 + math.max(depth(t.left), depth(t.right)) end
print(depth(root), root.left.right.left.key)

-- doubly linked list with a sentinel
local function dnode(v) return {value = v, prev = nil, next = nil} end
local head = dnode("head")
head.prev, head.next = head, head
local function push_back(v)
  local nd = dnode(v)
  nd.prev, nd.next = head.prev, head
  head.prev.next = nd
  head.prev = nd
  return nd
end
local a = push_back("a")
local b = push_back("b")
push_back("c")
b.prev.next, b.next.prev = b.next, b.prev
local fwd, back = {}, {}
local q = head.next
while q ~= head do fwd[#fwd + 1] = q.value; q = q.next end
q = head.prev
while q ~= head do back[#back + 1] = q.value; q = q.prev end
print(table.concat(fwd, ","), table.concat(back, ","), a.next.value, a.prev == head)

-- the same through a class with a metatable
local Node = {}
Node.__index = Node
function Node.new(v, nxt) return setmetatable({v = v, nxt = nxt}, Node) end
function Node:len() local c = 0; local s = self; while s do c = c + 1; s = s.nxt end; return c end
local tail = Node.new(1, nil)
local chain = Node.new(2, Node.new(3, tail))
print(chain:len(), chain.nxt.nxt == tail, tail.nxt)
