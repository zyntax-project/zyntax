# Sets

A set here iterates, prints and pops in the order its values were
added. CPython's order is its hash table's slot order instead, which for
small ints is ascending (`print({3, 1, 2})` shows `{1, 2, 3}` there and
`{3, 1, 2}` here). Cases print a set through `sorted`, or print only what
does not depend on the order, so their pinned output holds on both.
