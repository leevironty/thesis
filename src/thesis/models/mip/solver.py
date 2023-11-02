import pulp


def get_solver(threads: int, time_limit: int, output: bool = True, log_path: str | None = None, rel_gap: float = 0) -> pulp.LpSolver:
    solver = pulp.getSolver(
        'GUROBI',
        gapRel=rel_gap,
        logPath=log_path,
        threads=threads,
        timeLimit=time_limit,
        msg=output,
    )
    return solver
