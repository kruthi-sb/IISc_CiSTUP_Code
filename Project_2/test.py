"""
Author:  Kruthi S B
Status:  Complete
Created:  24-03-2024
Completed: 25-03-2024
Description: The goal of the assignment is to understand the Sifting Method from the paper and then implement it. This method can solve a large linear program by using a subset of columns.
"""

import cplex
from cplex.exceptions import CplexError

def read_LP(filename):
    """Reads an LP file and returns the CPLEX problem object."""
    prob = cplex.Cplex(filename)
    return prob

def solve_LP(prob):
    """Solves the given LP problem using the Sifting Method."""
    try:
        prob.solve()
    except CplexError as exc:
        print(exc)
        return

    # Solution status information
    status = prob.solution.get_status()
    print("Solution status:", prob.solution.status[status])
    print("Objective value:", prob.solution.get_objective_value())
    print()

def split_problem_and_solve(filename, num_columns=2000):
    """Splits the LP problem into subproblems and solves them."""
    prob = read_LP(filename)
    num_variables = prob.variables.get_num()
    num_iterations = (num_variables + num_columns - 1) // num_columns
    
    print("Splitting LP into", num_iterations, "subproblems...")
    for i in range(num_iterations):
        start_col = i * num_columns
        end_col = min((i + 1) * num_columns, num_variables)
        sub_filename = f"{filename[:-3]}_{i}.lp"
        sub_prob = cplex.Cplex()

        # Set problem type
        sub_prob.set_problem_type(prob.get_problem_type())

        # Add variables
        obj = prob.objective.get_linear()[start_col:end_col]
        lb = prob.variables.get_lower_bounds()[start_col:end_col]
        ub = prob.variables.get_upper_bounds()[start_col:end_col]
        sub_prob.variables.add(obj=obj, lb=lb, ub=ub)

        # Add constraints
        for j in range(prob.linear_constraints.get_num()):
            row = prob.linear_constraints.get_rows(j)
            senses = prob.linear_constraints.get_senses(j)
            rhs = prob.linear_constraints.get_rhs(j)
            lin_expr = [(row.ind[k] - start_col, row.val[k]) for k in range(len(row.ind)) if start_col <= row.ind[k] < end_col]
            sub_prob.linear_constraints.add(lin_expr=[lin_expr], senses=[senses], rhs=[rhs])

        sub_prob.write(sub_filename)
        print(f"Solving subproblem {i+1}/{num_iterations}...")
        solve_LP(sub_prob)

if __name__ == "__main__":
    filename = "./test1.lp"  # Relative path to the LP file
    split_problem_and_solve(filename)