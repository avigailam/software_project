import sys
import numpy as np
import symnmf


def error_handler():
    """
    Handles errors by printing an error message and exiting.
    
    """
    print(f"An Error Has Occurred")
    sys.exit(1)


def print_matrix(matrix):
    """
    Prints a matrix with values formatted to 4 decimal places.
    
    Args:
        matrix: A list of lists representing a matrix
    """
    for row in matrix:
        print(','.join('%.4f' % val for val in row))


def read_input(file_name):
    """
    Reads input from a text file and returns n, d, and the data matrix.
    
    Args:
        file_name: Path to the input text file
        
    Returns:
        Tuple of (n, d, data_points) where:
        - n: number of data points (rows)
        - d: number of dimensions (columns)
        - data_points: list of lists representing the data matrix
    """
    data_points = []
    
    try:
        with open(file_name, 'r') as f:
            lines = f.readlines()
            
            if not lines:
                error_handler()
            
            for line in lines:
                line = line.strip()
                if line:
                    try:
                        row = [float(val) for val in line.split(',')]
                        if not row:
                            error_handler()
                        data_points.append(row)
                    except ValueError:
                        error_handler()
        
        if not data_points:
            error_handler()
        
        n = len(data_points)
        d = len(data_points[0])
        
        # Validate that all rows have the same number of dimensions
        for i, row in enumerate(data_points):
            if len(row) != d:
                error_handler()
        
        return n, d, data_points
    
    except FileNotFoundError:
        error_handler()
    except IOError:
        error_handler()



def main():
    """
    Main function to handle command line arguments and execute the appropriate goal.
    """
    # Parse command line arguments
    if len(sys.argv) < 4:
        error_handler()
    
    try:
        k = int(sys.argv[1])
    except ValueError:
        error_handler()
    
    goal = sys.argv[2]
    file_name = sys.argv[3]
    
    # Read input file
    n, d, data_points = read_input(file_name)
    
    # Validate k
    if k <= 0 or k >= n:
        error_handler()
    
    # Validate goal
    valid_goals = ['sym', 'ddg', 'norm', 'symnmf']
    if goal not in valid_goals:
        error_handler()
    
    try:
        if goal == 'sym':
            # Calculate and output similarity matrix
            result = symnmf.sym(data_points, n, d)
            print_matrix(result)
        
        elif goal == 'ddg':
            # Calculate and output diagonal degree matrix
            result = symnmf.ddg(data_points, n, d)
            print_matrix(result)
        
        elif goal == 'norm':
            # Calculate and output normalized similarity matrix
            result = symnmf.norm(data_points, n, d)
            print_matrix(result)
        
        elif goal == 'symnmf':
            # Calculate W (normalized similarity matrix)
            W = symnmf.norm(data_points, n, d)
            
            # Initialize H using np.random.uniform()
            np.random.seed(1234)
            H = np.random.uniform(0, 1, (n, k)).tolist()
            
            # Optimize H using the C extension
            result = symnmf.symnmf(W, H, n, k)
            print_matrix(result)
    
    except Exception as e:
        error_handler()



if __name__ == '__main__':
    main()
