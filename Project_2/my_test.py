import cplex
from cplex.exceptions import CplexError

def solve_lp_from_lp_file(lp_file):
    try:
        # Create a CPLEX problem object
        prob = cplex.Cplex(lp_file)
        
        # Solve the problem
        prob.solve()
        
        # Print solution status
        print("Solution status:", prob.solution.get_status())
        
        # Print objective value
        print("Objective value:", prob.solution.get_objective_value())
        
        # Print variable values
        num_vars = prob.variables.get_num()
        for i in range(num_vars):
            var_name = prob.variables.get_names()[i]
            var_value = prob.solution.get_values(i)
            print(f"{var_name} = {var_value}")
            
    except CplexError as exc:
        print(exc)

if __name__ == "__main__":
    lp_file = "my_test.lp"  # Replace with the path to your .lp file
    solve_lp_from_lp_file(lp_file)

