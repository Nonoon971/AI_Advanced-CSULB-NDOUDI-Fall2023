# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 02:28:48 2023

@author: CYBER19
"""

import random
import numpy as np
import matplotlib.pyplot as plt

num_city = 100
num_air = 3
num_center = 5
sigma = 0.1
cities = set()
airports = []

#A random initial state
for i in range(num_center):
    x = random.random()
    y = random.random()
    xc = np.random.normal(x, sigma, num_city//num_center)
    yc = np.random.normal(y, sigma, num_city//num_center)
    cities = cities.union(zip(xc, yc))


for i in range(num_air):
    x = random.random()
    y = random.random()
    airports.append((x,y)) 


zip_cities = zip(*cities)
plt.scatter(*zip_cities, marker='+',color='b', label='Cities')
zip_airs = zip(*airports)
plt.scatter(*zip_airs, marker='*', color='r', s=100, label='Airports')
plt.legend()
plt.title('initial data')
plt.show()

print("INITIAL AIRPORTS COORDINATES")
for i, (x, y) in enumerate(airports):
    print(f'airports {i + 1}: COORDINATES ({x:.2f}, {y:.2f})')

#cities whose closest airport is airport i
def closest_cities_to_airports(cities, airports):
    closestCities_airports = [[] for k in range(num_air)]
    for city_x, city_y in cities:
        #Initalization of the closest_airport index and the min distance for comparison
        closest_airport = None
        min_distance = float('inf')
        for i, (airport_x, airport_y) in enumerate(airports):
            #Calculation of the distance (x1 -x2)² + (y1 - y2)²
            distance = (airport_x - city_x) ** 2 + (airport_y - city_y) ** 2
            #Compare the distance of the city and each airport to see the closest one
            if distance < min_distance:
                min_distance = distance
                closest_airport = i
                
        closestCities_airports[closest_airport].append((city_x, city_y))
    return closestCities_airports

def objective_function(closestCity):
    sumTotal = 0
    # n is the number of the airports
    for n in range(num_air):
        closest_cities_to_airport_n = closestCity[n]
        #coordinates of cities whose closest airport is airport n.
        for city_x, city_y in closest_cities_to_airport_n:
            distance = (airports[n][0] - city_x) ** 2 + (airports[n][1] - city_y) ** 2
            sumTotal += distance
    return sumTotal


alpha = 0.01
# Number of iterations chosen where we appear to have an optimal solution for the airport coordinates
num_iterations = 50
closestCity = closest_cities_to_airports(cities, airports)

# List to store the values of the objective function at each iteration
objective_values = []

# gradient descent to minimize the objective funtion
for iteration in range(num_iterations):
    gradient = [(0, 0) for k in range(num_air)]
    for i in range(num_air):
        for city_x, city_y in closestCity[i]:
            #gradient calculation with the derivation according to xi and yi
            gradient[i] = (gradient[i][0] + 2 * (airports[i][0] - city_x), gradient[i][1] + 2 * (airports[i][1] - city_y))
    
    #Airports coordinates update
    airports = [(airports[i][0] - alpha * gradient[i][0], airports[i][1] - alpha * gradient[i][1]) for i in range(num_air)]
    
    # Calculation of the objective function value for this iteration and add it to the list
    objective_values.append(objective_function(closestCity))

    
# Affichage des résultats
zip_cities = zip(*cities)
plt.scatter(*zip_cities, marker='+', color='b', label='Cities')
zip_airs = zip(*airports)
plt.scatter(*zip_airs, marker='*', color='r', s=100, label='Airports')
plt.legend()
plt.title('Results after optimization')
plt.show()

# Affichage des coordonnées finales des aéroports
for i, (x, y) in enumerate(airports):
    print(f'airports {i + 1}: COORDINATES ({x:.2f}, {y:.2f})')
    
# Display of the objective function values over iterations
plt.scatter(range(num_iterations), objective_values)
plt.xlabel('Iteration')
plt.ylabel('Objective function value')
plt.title('Evolution of the objective function through iterations')
plt.show()



    
