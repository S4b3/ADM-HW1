# Say "Hello, World!" With Python

from __future__ import print_function
import functools
if __name__ == '__main__':
    print("Hello, World!")

# Python If-Else

#!/bin/python

import math
import os
import random
import re
import sys

if __name__ == '__main__':
    n = int(raw_input().strip())

    # Limit case
    if n > 100 or n < 1:
        sys.exit()
    # Odd
    if (n % 2 != 0):
        print("Weird")
    # Even
    elif (n % 2 == 0):
        if(n >= 2 and n <= 5):
            print("Not Weird")
        elif (n >= 6 and n <= 20):
            print("Weird")
        else:
            print("Not Weird")

# Arithmetic Operators

if __name__ == '__main__':
    a = int(raw_input())
    b = int(raw_input())

    ceil = 10 ** 10
    if not (a >= 1 and a <= ceil and b >= 1 and b <= ceil):
        sys.exit()
    print(a + b)
    print(a - b)
    print(a * b)

# Python: Division

from __future__ import division

if __name__ == '__main__':
    a = int(raw_input())
    b = int(raw_input())

    print(a//b)
    print(a/b)

# Loops

if __name__ == '__main__':
    n = int(raw_input())

    for i in range(n):
        if(i >= 0):
            print(i**2)

# Write a function


def is_leap(year):
    leap = False
    ceil = 10 ** 5
    # Limit case
    if(year < 1900 or year > ceil):
        return
    # Leap year check
    if (year % 4 == 0):
        if (year % 100 == 0):
            leap = year % 400 == 0
        else:
            leap = True
    return leap


year = int(raw_input())
print is_leap(year)

# Print Function

if __name__ == '__main__':
    n = int(raw_input())
    if(n < 1 or n > 150):
        sys.exit()
    for i in xrange(1, n+1):
        print(i, end='')

# List Comprehensions

if __name__ == '__main__':
    x = int(raw_input())
    y = int(raw_input())
    z = int(raw_input())
    n = int(raw_input())

    output = []
    for i in range(x+1):
        for j in range(y+1):
            for k in range(z+1):
                if(i + j + k != n):
                    output.append([i, j, k])

    print(output)

# Find the Runner-Up Score!


if __name__ == '__main__':
    n = int(raw_input())
    arr = map(int, raw_input().split())

    arr.sort(reverse=True)
    for i in arr:
        if(arr.index(i) != 0 and i != arr[0]):
            print(i)
            sys.exit()

# Nested Lists


def useSecondElementInArray(e):
    return e[1]


def useFirstElementInArray(e):
    return e[0]


if __name__ == '__main__':
    students = []
    for _ in range(int(raw_input())):
        name = raw_input()
        score = float(raw_input())
        students.append([name, score])
    # sort input
    students.sort(key=useSecondElementInArray)
    output = []
    # define a support variable to avoid checking length at every iteration
    populated = False

    for s in students:
        if((not populated) and students.index(s) != 0 and useSecondElementInArray(s) != useSecondElementInArray(students[0])):
            output.append(s)
            populated = True
        elif (populated and students.index(s) != 0 and useSecondElementInArray(s) == useSecondElementInArray(output[0])):
            output.append(s)

    output.sort(key=useFirstElementInArray)

    for os in output:
        print(useFirstElementInArray(os))

# Finding the percentage


def sum(a, b):
    return a+b


if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()

    currentGrades = student_marks[query_name]

    average = functools.reduce(sum, currentGrades) / len(currentGrades)

    print(format(average, ".2f"))

# Lists

if __name__ == '__main__':
    N = int(input())
    output = []

    for n in range(N):
        commands = (input().split(' '))
        if(commands[0] == 'insert'):
            output.insert(int(commands[1]), int(commands[2]))
        elif(commands[0] == 'print'):
            print(output)
        elif(commands[0] == 'remove'):
            output.remove(int(commands[1]))
        elif(commands[0] == 'append'):
            output.append(int(commands[1]))
        elif(commands[0] == 'sort'):
            output.sort()
        elif(commands[0] == 'pop'):
            output.pop()
        elif(commands[0] == 'reverse'):
            output.reverse()

# Tuples

if __name__ == '__main__':
    n = int(raw_input())
    integer_list = map(int, raw_input().split())
    tpl = tuple(integer_list)
    print hash(tpl)


# sWAP cASE

def swap_case(s):
    output = ''
    for char in s:
        if(char.islower()):
            output = output + char.upper()
        else:
            output = output + char.lower()
    return output


if __name__ == '__main__':
    s = raw_input()
    result = swap_case(s)
    print result


# String Split and Join

def split_and_join(line):
    return "-".join(line.split(' '))


if __name__ == '__main__':
    line = raw_input()
    result = split_and_join(line)
    print result


# What's Your Name?

def print_full_name(first, last):
    print 'Hello {} {}! You just delved into python.'.format(first, last)


if __name__ == '__main__':
    first_name = raw_input()
    last_name = raw_input()
    print_full_name(first_name, last_name)

# Mutations


def mutate_string(string, position, character):
    return string[:position] + character + string[(position+1):]


if __name__ == '__main__':
    s = input()
    i, c = input().split()
    s_new = mutate_string(s, int(i), c)
    print(s_new)

# Find a string


def count_substring(string, sub_string):
    control = True
    count = start = 0
    while(control):
        start = string.find(sub_string, start) + 1
        if(start > 0):
            count += 1
        else:
            control = False
    return count


if __name__ == '__main__':
    string = input().strip()
    sub_string = input().strip()

    count = count_substring(string, sub_string)
    print(count)
