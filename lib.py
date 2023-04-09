print("lib.py is my library file where I store my functions")

import random

def Monte_Carlo_pi(n):
    num_points_circle = 0
    num_points_total = 0
    for _ in range(n):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        distance = x**2 + y**2
        if distance <= 1:
            num_points_circle += 1
        num_points_total += 1
    return 4 * num_points_circle / num_points_total

def main():
    print(Monte_Carlo_pi(1000000))

if __name__ == '__main__':
    main()
elif __name__ == 'lib':
    print("Executing lib.py from",__name__)


#print(__name__)