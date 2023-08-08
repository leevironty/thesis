import pulp


def get_solver(log_path: str, threads: int, time_limit: int) -> pulp.LpSolver:
    solver = pulp.getSolver(
        'GUROBI',
        gapRel=0,
        logPath=log_path,
        threads=threads,
        timeLimit=time_limit,
    )
    return solver
