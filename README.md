# Project Summary
## This project compares three global optimization algorithms:

TIAM: Uses interval arithmetic and multithreading for concurrent interval processing.
IAG: Utilizes gradient information and multithreading.
IAOICT: Focuses on interval arithmetic without multithreading.
## Key Components
Libraries: numpy, mpmath, scipy.optimize, concurrent.futures, pandas, time, re.
Test Functions: Rastrigin, Rosenbrock, Quadratic, Linear, Third-degree, and Fourth-degree polynomials.
Data Analysis: Results stored in Pandas DataFrames, including best x, f(x), NFE, initial bounds, and computation time.
## Conclusion
The project evaluates the performance and efficiency of the three optimization methods across various functions and bounds.
