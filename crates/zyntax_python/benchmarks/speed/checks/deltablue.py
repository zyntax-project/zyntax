import deltablue
deltablue.chain_test(50)
deltablue.projection_test(50)
p = deltablue.planner
print(p is None)
print(p.current_mark, len(p.make_plan(deltablue.OrderedCollection())))
v = deltablue.Variable("x", 7)
print(v, v.stay, v.walk_strength.name, deltablue.Strength.NORMAL.next_weaker().name)
print(deltablue.Strength.stronger(deltablue.Strength.REQUIRED, deltablue.Strength.WEAKEST), deltablue.Strength.weakest_of(deltablue.Strength.NORMAL, deltablue.Strength.WEAK_DEFAULT).name)
print("done")
