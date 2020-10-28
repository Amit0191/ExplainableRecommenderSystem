from scipy.optimize import linprog
from pulp import *
from Recommender import *

model = LpProblem(name="GameTheory_Recommender", sense=LpMaximize)
x1 = LpVariable(name="x1", lowBound=0)
x2 = LpVariable(name="x2", lowBound=0)
x3 = LpVariable(name="x3", lowBound=0)
x4 = LpVariable(name="x4", lowBound=0)
x5 = LpVariable(name="x5", lowBound=0)
x6 = LpVariable(name="x6", lowBound=0)
x7 = LpVariable(name="x7", lowBound=0)
x8 = LpVariable(name="x8", lowBound=0)
x9 = LpVariable(name="x9", lowBound=0)
x10 = LpVariable(name="x10", lowBound=0)
x11 = LpVariable(name="x11", lowBound=0)
x12 = LpVariable(name="x12", lowBound=0)
x13 = LpVariable(name="x13", lowBound=0)
x14 = LpVariable(name="x14", lowBound=0)
x15 = LpVariable(name="x15", lowBound=0)
x16 = LpVariable(name="x16", lowBound=0)
x17 = LpVariable(name="x17", lowBound=0)
x18 = LpVariable(name="x18", lowBound=0)
x19 = LpVariable(name="x19", lowBound=0)
x20 = LpVariable(name="x20", lowBound=0.20)
w = LpVariable(name="w")

model += (x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 +
          x11 + x12 + x13 + x14 + x15 + x16 + x17 + x18 + x19 + x20 == 1, "probability_constraint")


varList = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10',
           'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20']

varList2 = np.array([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10,
                     x11, x12, x13, x14, x15, x16, x17, x18, x19, x20])

# model += (4 * x - 5 * y >= -10, "blue_constraint")
# model += (-x + 2 * y >= -2, "yellow_constraint")
# model += (-x + 5 * y == 15, "green_constraint")
# obj_func = x + 2 * y
# model += obj_func
# model += lpSum([x, 2 * y])
# status = model.solve()

model += w

for i in range(len(M_for_user)):
    row = np.array(M_for_user[i, :])
    constraint_v = varList2.dot(row)
    model += (w - constraint_v <= 0)

status = model.solve()
print(model)
print(f"status: {model.status}, {LpStatus[model.status]}")
print(f"objective: {model.objective.value()}")
for var in model.variables():
    print(f"{var.name}: {var.value()}")


