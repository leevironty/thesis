import pulp


def get_solver() -> pulp.LpSolver:
    solver = pulp.getSolver('GUROBI', gapRel=0.00001)
    return solver
