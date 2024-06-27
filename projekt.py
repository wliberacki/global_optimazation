#!/usr/bin/env python
# coding: utf-8

# ### Importowanie potrzebnych bibliotek

# In[1]:


#Potrzebne biblioteki
import numpy as np
from mpmath import iv
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import time
import re
from tabulate import tabulate


# In[2]:


# Obliczanie gradientu funkcji
def gradient(f, x):
    grad = np.zeros_like(x)
    h = 1e-8
    for i in range(len(x)):
        x1 = x.copy()
        x2 = x.copy()
        x1[i] -= h / 2
        x2[i] += h / 2
        grad[i] = (f(x2) - f(x1)) / h
    return grad


# ### Implementacja metody TIAM

# In[3]:


# Funkcja pomocnicza do przetwarzania przedziału (używana w TIAM)
def process_interval(f, intervals, NFE):
    mid = [float(i.mid) for i in intervals]
    res = minimize(f, mid, bounds=[(float(i.a), float(i.b)) for i in intervals], method='L-BFGS-B')
    NFE[0] += res.nfev
    return res.x, res.fun

# Traditional Interval analysis global minimization Algorithm with Monotonicity test (TIAM)
def tiam_method(f, bounds, tol=1e-6, max_iter=100):
    start = time.time()
    best_x = None
    best_f = float('inf')
    NFE = [0]
    intervals = [bounds]
    
    with ThreadPoolExecutor() as executor:
        for _ in range(max_iter):
            new_intervals = []
            futures = {executor.submit(process_interval, f, interval, NFE): interval for interval in intervals}
            
            for future in as_completed(futures):
                x, fun = future.result()
                if fun < best_f:
                    best_f = fun
                    best_x = x
                
                interval = futures[future]
                mid = [float(i.mid) for i in interval]
                
                if all([(float(i.b) - float(i.a)) < tol for i in interval]):
                    continue
                
                for i in range(len(interval)):
                    left = interval.copy()
                    right = interval.copy()
                    left[i] = iv.mpf([left[i].a, mid[i]])
                    right[i] = iv.mpf([mid[i], right[i].b])
                    new_intervals.extend([left, right])
            
            intervals = new_intervals
            if not intervals:
                break
            
            if len(new_intervals) > 0 and abs(best_f - fun) < tol:
                break
    time_elapsed = time.time() - start
    return best_x, best_f, NFE[0], bounds, time_elapsed


# ### Implementacja metody IAG

# In[4]:


# Funkcja pomocnicza do przetwarzania przedziału z gradientem (używana w IAG)
def process_interval_with_gradient(f, intervals, NFE):
    mid = [float(i.mid) for i in intervals]
    res = minimize(f, mid, bounds=[(float(i.a), float(i.b)) for i in intervals], method='L-BFGS-B', jac=lambda x: gradient(f, x))
    NFE[0] += res.nfev
    grad = gradient(f, mid)
    return res.x, res.fun, grad

# Interval analysis global minimization Algorithm using Gradient information (IAG)
def iag_method(f, bounds, tol=1e-6, max_iter=100):
    start = time.time()
    best_x = None
    best_f = float('inf')
    NFE = [0]
    intervals = [bounds]
    
    with ThreadPoolExecutor() as executor:
        for _ in range(max_iter):
            new_intervals = []
            futures = {executor.submit(process_interval_with_gradient, f, interval, NFE): interval for interval in intervals}
            
            for future in as_completed(futures):
                x, fun, grad = future.result()
                if fun < best_f:
                    best_f = fun
                    best_x = x
                
                interval = futures[future]
                mid = [float(i.mid) for i in interval]
                
                if all([(float(i.b) - float(i.a)) < tol for i in interval]):
                    continue
                
                for i in range(len(interval)):
                    left = interval.copy()
                    right = interval.copy()
                    left[i] = iv.mpf([left[i].a, mid[i]])
                    right[i] = iv.mpf([mid[i], right[i].b])
                    new_intervals.extend([left, right])
            
            intervals = new_intervals
            if not intervals:
                break
            
            if len(new_intervals) > 0 and abs(best_f - fun) < tol:
                break
    time_elapsed = time.time() - start        
    return best_x, best_f, NFE[0], bounds, time_elapsed


# ### Implementacja metody IAOICT

# In[5]:


# Interval Arithmetic Oriented Interval Computing Technique (IAOICT)
def iaoict_method(f, bounds, tol=1e-6, max_iter=100):
    start = time.time()
    best_x = None
    best_f = float('inf')
    NFE = [0]
    intervals = [bounds]
    
    while intervals and max_iter > 0:
        interval = intervals.pop(0)
        mid = [float(i.mid) for i in interval]
        res = minimize(f, mid, bounds=[(float(i.a), float(i.b)) for i in interval], method='L-BFGS-B')
        NFE[0] += res.nfev
        
        if res.fun < best_f:
            best_f = res.fun
            best_x = res.x
        
        mid = [float(i.mid) for i in interval]
        if all([(float(i.b) - float(i.a)) < tol for i in interval]):
            continue
        
        for i in range(len(interval)):
            left = interval.copy()
            right = interval.copy()
            left[i] = iv.mpf([left[i].a, mid[i]])
            right[i] = iv.mpf([mid[i], right[i].b])
            intervals.extend([left, right])
        
        max_iter -= 1
    time_elapsed = time.time() - start  
    return best_x, best_f, NFE[0], bounds, time_elapsed


# ### Definiowanie funkcji 

# In[6]:


def rastrigin(x):
    A = 10
    return A * len(x) + sum([(xi**2 - A * np.cos(2 * np.pi * float(xi))) for xi in x])

def rosenbrock(x):
    return sum([100 * (float(x[i+1]) - float(x[i])**2)**2 + (1 - float(x[i]))**2 for i in range(len(x)-1)])

def quadratic(x):
    return sum([float(xi)**2 -4*xi + 5 for xi in x])

def linear(x):
    return sum([xi for xi in x])

def third_degree_polynomial(x):
    return sum([pow(xi,3) - 7*pow(xi, 2) + 4*xi for xi in x])

def fourth_degree_polynomial(x):
    return sum([2*pow(xi,4) - 9*pow(xi,3) +3*pow(xi, 2) - 2*xi + 7 for xi in x])


# ### Testy

# In[7]:


def interval_to_string(i):
    return f"{i[0]}"


# In[8]:


# Lista funkcji testowych
test_functions = [rastrigin, rosenbrock, quadratic, linear, third_degree_polynomial, fourth_degree_polynomial]
function_names = ["Rastrigin", "Rosenbrock", "Quadratic", "Linear", "Third degree polynomial", "Fourth degree polynomial"]

list_of_bounds = [
    [iv.mpf([2, 7])],
    [iv.mpf([-1, 1])],
    [iv.mpf([-13, -4])],
    [iv.mpf([-5, 9])],
    [iv.mpf([-10000, 10000])]
]

TIAM_results = []
IAG_results = []
IAOICT_results = []

for bounds in list_of_bounds:
    results_tiam = {}
    results_iag = {}
    results_iaoict = {}

    # Przeprowadzenie eksperymentów dla TIAM
    for func, name in zip(test_functions, function_names):
        best_x, best_f, nfe, initial_bounds, time_elapsed = tiam_method(func, bounds)
        results_tiam[name] = (best_x, best_f, nfe, initial_bounds, time_elapsed)

    # Przeprowadzenie eksperymentów dla IAG
    for func, name in zip(test_functions, function_names):
        best_x, best_f, nfe, initial_bounds, time_elapsed = iag_method(func, bounds)
        results_iag[name] = (best_x, best_f, nfe, initial_bounds, time_elapsed)

    # Przeprowadzenie eksperymentów dla IAOICT
    for func, name in zip(test_functions, function_names):
        best_x, best_f, nfe, initial_bounds, time_elapsed = iaoict_method(func, bounds)
        results_iaoict[name] = (best_x, best_f, nfe, initial_bounds, time_elapsed)

    for name, result in results_tiam.items():
        best_x, best_f, nfe, initial_bounds, time_elapsed = result
        readable_bounds = [(float(b.a), float(b.b)) for b in initial_bounds]
        TIAM_record = {"function": name, "x_value": best_x[0], "y_value": best_f, "nfe": nfe, 
                       'interval': interval_to_string(readable_bounds), "time_elapsed": time_elapsed}
        TIAM_results.append(TIAM_record)
        
    for name, result in results_iag.items():
        best_x, best_f, nfe, initial_bounds, time_elapsed = result
        readable_bounds = [(float(b.a), float(b.b)) for b in initial_bounds]
        IAG_record = {"function": name, "x_value": best_x[0], "y_value": best_f, "nfe": nfe, 
                       'interval': interval_to_string(readable_bounds), "time_elapsed": time_elapsed}
        IAG_results.append(IAG_record)
        
    for name, result in results_iaoict.items():
        best_x, best_f, nfe, initial_bounds, time_elapsed = result
        readable_bounds = [(float(b.a), float(b.b)) for b in initial_bounds]
        IAOICT_record = {"function": name, "x_value": best_x[0], "y_value": best_f, "nfe": nfe, 
                       'interval': interval_to_string(readable_bounds), "time_elapsed": time_elapsed}
        IAOICT_results.append(IAOICT_record)


# In[9]:


TIAM_df = pd.DataFrame(TIAM_results).set_index("function")
IAG_df = pd.DataFrame(IAG_results).set_index("function")
IAOICT_df = pd.DataFrame(IAOICT_results).set_index("function")


# In[10]:


print(tabulate(TIAM_df, headers='keys', tablefmt='psql'))


# In[11]:


print(tabulate(IAG_df, headers='keys', tablefmt='psql'))


# In[12]:


print(tabulate(IAOICT_df, headers='keys', tablefmt='psql'))


# In[13]:


cols = ['method', 'iterations', 'time_elapsed']
names = ['TIAM', 'IAG', 'IAOICT']
iterations = [TIAM_df['nfe'].sum(), IAG_df['nfe'].sum(), IAOICT_df['nfe'].sum()]
times = [TIAM_df['time_elapsed'].sum(), IAG_df['time_elapsed'].sum(), IAOICT_df['time_elapsed'].sum()]


# In[14]:


comparison = pd.DataFrame(list(zip(names, iterations, times)), columns = cols).set_index('method')


# In[15]:


print(tabulate(comparison, headers='keys', tablefmt='psql'))


# In[16]:


IAOICT_df_nit = IAOICT_df[['nfe','time_elapsed','interval']].copy()


# In[17]:


def get_interval_length(i):
    LENGTHS = {'(2.0, 7.0)': 5,'(-1.0, 1.0)': 2, '(-13.0, -4.0)':9, '(-5.0, 9.0)': 14 , '(-10000.0, 10000.0)': 20000}
    return LENGTHS[i]


# In[18]:


IAOICT_df_nit['interval_length'] = IAOICT_df_nit['interval'].map(get_interval_length)


# In[19]:


IAOICT_df_nit = IAOICT_df_nit.drop(['interval'], axis = 1)


# In[20]:


print(tabulate(IAOICT_df_nit.head(10), headers='keys', tablefmt='psql'))


# In[21]:


print(tabulate(IAOICT_df_nit.corr(), headers='keys', tablefmt='psql'))

