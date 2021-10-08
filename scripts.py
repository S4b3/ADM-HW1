# Say "Hello, World!" With Python

# from __future__ import print_function
import operator
import xml.etree.ElementTree as etree
from html.parser import HTMLParser
import email.utils
from collections import deque
from collections import Counter, OrderedDict
import datetime
import calendar
from collections import namedtuple
from collections import defaultdict
from collections import Counter
import textwrap
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
    n = int(input().strip())

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
    a = int(input())
    b = int(input())

    ceil = 10 ** 10
    if not (a >= 1 and a <= ceil and b >= 1 and b <= ceil):
        sys.exit()
    print(a + b)
    print(a - b)
    print(a * b)

# Python: Division

# from __future__ import division

if __name__ == '__main__':
    a = int(input())
    b = int(input())

    print(a//b)
    print(a/b)

# Loops

if __name__ == '__main__':
    n = int(input())

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


year = int(input())
print(is_leap(year))

# Print Function

if __name__ == '__main__':
    n = int(input())
    if(n < 1 or n > 150):
        sys.exit()
    for i in range(1, n+1):
        print(i, end='')

# List Comprehensions

if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())

    output = []
    for i in range(x+1):
        for j in range(y+1):
            for k in range(z+1):
                if(i + j + k != n):
                    output.append([i, j, k])

    print(output)

# Find the Runner-Up Score!


if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())

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
    for _ in range(int(input())):
        name = input()
        score = float(input())
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
    n = int(input())
    integer_list = map(int, input().split())
    tpl = tuple(integer_list)
    print(hash(tpl))


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
    s = input()
    result = swap_case(s)
    print(result)


# String Split and Join

def split_and_join(line):
    return "-".join(line.split(' '))


if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)


# What's Your Name?

def print_full_name(first, last):
    print('Hello {} {}! You just delved into python.'.format(first, last))


if __name__ == '__main__':
    first_name = input()
    last_name = input()
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

# String Validators

if __name__ == '__main__':
    s = input()
    print(len([char for char in s if char.isalnum()]) > 0)
    print(len([char for char in s if char.isalpha()]) > 0)
    print(len([char for char in s if char.isdigit()]) > 0)
    print(len([char for char in s if char.islower()]) > 0)
    print(len([char for char in s if char.isupper()]) > 0)

# Text Alignment


thickness = int(input())  # This must be an odd number
c = 'H'

# Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

# Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

# Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))

# Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

# Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c +
          (c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))

# Text Wrap


def wrap(string, max_width):

    return "\n".join([string[i:i+max_width] for i in range(0, len(string), max_width)])


if __name__ == '__main__':
    string, max_width = input(), int(input())
    result = wrap(string, max_width)
    print(result)

# Designer Door Mat

n, m = map(int, input().split())

for i in range(1, n, 2):
    print((i * ".|.").center(m, "-"))
print("WELCOME".center(m, "-"))
for i in range(n-2, -1, -2):
    print((i * ".|.").center(m, "-"))

# String Formatting


def print_formatted(number):
    # your code goes here
    width = len(format(number, 'b'))
    for i in range(1, n+1):
        print("{0:{width}d} {0:{width}o} {0:{width}X} {0:{width}b}".format(
            i, width=width))


if __name__ == '__main__':
    n = int(input())
    print_formatted(n)

# Alphabet Rangoli


def print_rangoli(size):
    last_letter = chr(size)
    rows = []
    for i in range(1, n+1):
        row = []
        for j in range(0, i):
            row.append(chr(ord('a') + n-j - 1))
        rowReverse = list(row)
        rowReverse.reverse()
        ''' 9 elementi vuol dire [(5*2) - 1] + ((5*2 - 1) / 2) '''
        amount = ((size * 2) - 1) + ((size * 2) - 2)
        output = '-'.join(row + rowReverse[1:]).center(amount, '-')
        rows.append(output)
        print(output)

    rows.reverse()
    for row in rows[1:]:
        print(row)


if __name__ == '__main__':
    n = int(input())
    print_rangoli(n)

# Capitalize!

#!/bin/python3


# Complete the solve function below.

def solve(s):
    return ' '.join([s.capitalize() for s in s.split(' ')])


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    s = input()

    result = solve(s)

    fptr.write(result + '\n')

    fptr.close()

# The Minion Game


def minion_game(string):
    vowels = 'AEIOU'
    kevin = 0
    stuart = 0
    for i in range(len(s)):
        if s[i] in vowels:
            kevin += (len(s)-i)
        else:
            stuart += (len(s)-i)
    if kevin > stuart:
        print("Kevin", kevin)
    elif kevin < stuart:
        print("Stuart", stuart)
    else:
        print("Draw")


if __name__ == '__main__':
    s = input()
    minion_game(s)

# Merge the Tools!


def merge_the_tools(string, k):
    splits = [string[i*k:(i+1)*k] for i in range(len(string)//k)]

    print('\n'.join(["".join(dict.fromkeys(s)) for s in splits]))


if __name__ == '__main__':
    string, k = input(), int(input())
    merge_the_tools(string, k)

# collections.Counter()


if __name__ == '__main__':
    shoesAmount = input()
    shoes = Counter(list(map(int, input().split())))
    moneyEarned = 0
    for i in range(0, int(input())):
        shoe, price = map(int, input().split())
        if(int(shoes[shoe]) > 0):
            shoes[shoe] -= 1
            moneyEarned += price
    print(moneyEarned)

# DefaultDict Tutorial


if(__name__ == '__main__'):
    n, m = map(int, input().split())
    dictionary = defaultdict(list)

    for i in range(0, n):
        dictionary[input()].append(i+1)
    for i in range(0, m):
        print(" ".join(map(str, dictionary[input()])) or '-1')

# Collections.namedtuple()

n = int(input())
Student = namedtuple('Student', map(str, input().split()))
studs = list([Student._make(map(str, input().split())) for i in range(0, n)])
print("{:.2f}".format(sum(int(s.MARKS) for s in studs) / len(studs)))

# Introduction to Sets


def average(array):
    hset = set(array)
    hsum = 0
    for el in hset:
        hsum += el
    return hsum / len(hset)


if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    result = average(arr)
    print(result)

# Symmetric Difference

n = int(input())
a = set(map(int, input().split()))
m = int(input())
b = set(map(int, input().split()))
dif = list(el for el in a.difference(b).union(b.difference(a)))
dif.sort()
for el in dif:
    print(str(el))

# No Idea!

n, m = input().split()
happy = input().split()
A = set(input().split())
B = set(input().split())
print(sum([(i in A) - (i in B) for i in happy]))

# Set .add()

output = set()
for i in range(0, int(input())):
    output.add(input())
print(len(output))

# Set .discard(), .remove() & .pop()

n = int(input())
s = set(map(int, input().split()))
for i in range(0, int(input())):
    command = input().split()
    if(command[0] == 'remove' and int(command[1]) in s):
        s.remove(int(command[1]))
    elif (command[0] == 'discard'):
        s.discard(int(command[1]))
    else:
        s.pop()
print(sum(s))

# Calendar Module

m, d, y = map(int, input().split())
print((calendar.day_name[calendar.weekday(y, m, d)]).upper())

# Time Delta
#!/bin/python3


def time_delta(t1, t2):
    d1 = datetime.datetime.strptime(t1, '%a %d %b %Y %H:%M:%S %z')
    d2 = datetime.datetime.strptime(t2, '%a %d %b %Y %H:%M:%S %z')
    return str(abs(int((d1 - d2).total_seconds())))


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        t1 = input()

        t2 = input()

        delta = time_delta(t1, t2)

        fptr.write(delta + '\n')

    fptr.close()

# Exceptions

n = int(input())

for i in range(0, n):
    try:
        a, b = map(int, input().split())
        print(a // b)
    except ZeroDivisionError as e:
        print("Error Code: integer division or modulo by zero")
    except ValueError as e:
        print("Error Code:", e)

# Collections.OrderedDict()

items = dict()
for i in range(int(input())):
    key, i, value = input().rpartition(" ")
    items[key] = items.get(key, 0) + int(value)
for k, v in items.items():
    print(k, v)

# Set .union() Operation

n = input()
a = set(input().split())
m = input()
b = set(input().split())
print(len(a.union(b)))

# Set .intersection() Operation

_, a = input(), set(input().split())
_, b = input(), set(input().split())
print(len(a.intersection(b)))

# Set .difference() Operation

_, a = input(), set(input().split())
_, b = input(), set(input().split())
print(len(a.difference(b)))

# Set .symmetric_difference() Operation

_, a = input(), set(input().split())
_, b = input(), set(input().split())
print(len(a.symmetric_difference(b)))

# Set Mutations

if __name__ == '__main__':
    (_, a) = (int(input()), set(map(int, input().split())))
    b = int(input())
    for _ in range(b):
        (command, newSet) = (input().split()
                             [0], set(map(int, input().split())))
        getattr(a, command)(newSet)

    print(sum(a))


# The Captain's Room

k, arr = int(input()), list(map(int, input().split()))
myset = set(arr)

print(((sum(myset)*k)-(sum(arr)))//(k-1))

# Check Subset


for i in range(0, int(input())):
    lenA, a, lenB, b = int(input()), list(map(int, input().split())), int(
        input()), list(map(int, input().split()))
    output = True
    for el in a:
        if not el in b:
            output = False
            break
    print(output)

# Check Strict Superset

a = set(input().split())
print(all([a.issuperset(set(input().split())) for _ in range(int(input()))]))

# Incorrect Regex

for i in range(int(input())):
    try:
        print(bool(re.compile(input())))
    except re.error:
        print('False')

# Word Order


class CounterDictionary(Counter, OrderedDict):
    pass


d = CounterDictionary(input() for _ in range(int(input())))
print(len(d))
print(" ".join(str(el) for el in d.values()))

# Collections.deque()


d = deque()
for _ in range(int(input())):
    func, *num = input().split()
    getattr(d, func)(*num)
print(' '.join(d))

# Company Logo

#!/bin/python3


class CounterDictionary(Counter, OrderedDict):
    pass


if __name__ == '__main__':
    s = input()
    letters = CounterDictionary(sorted(s)).most_common(3)
    [print(*letter) for letter in letters]

# Piling Up!


def highest(a, b):
    if a >= b:
        return a
    return b


def stackable(n, blocks):
    curr = highest(blocks[0], blocks[-1])
    for _ in range(0, n):
        el = highest(blocks[0], blocks[-1])
        if el > curr:
            return('No')
        elif el == blocks[0]:
            curr = blocks[0]
            blocks.popleft()
        else:
            curr = blocks[-1]
            blocks.pop()
    return('Yes')


for _ in range(0, int(input())):
    n = int(input())
    blocks = deque(map(int, input().split()))
    print(stackable(n, blocks))

# Zipped!

n, x = map(int, input().split())
s = [list(map(float, input().split())) for _ in range(x)]
out = list(zip(*s))
for i in range(n):
    print("%0.1f" % (sum(out[i])/x))


# Athlete Sort

#!/bin/python3


if __name__ == '__main__':
    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))

    k = int(input())

    for el in sorted(arr, key=lambda row: row[k]): print(*el)

# ginortS

s = input()
s = sorted(s, key=lambda char: (char.isdigit() and int(char) %
           2 == 0, char.isdigit(), char.isupper(), char.islower(), char))
print(*s, sep='')

# Map and Lambda Function


def cube(x): return x ** 3


def fibonacci(n):
    def fib(k): return k if k < 2 else fib(k - 1) + fib(k - 2)
    return list(map(fib, range(n)))


if __name__ == '__main__':
    n = int(input())
    print(list(map(cube, fibonacci(n))))

# Detect Floating Point Number

for _ in range(int(input())):
    print(bool(re.match(r'^[-+]?[0-9]*\.[0-9]+$', input())))

# Re.split()

regex_pattern = r"[.,]+"
print("\n".join(re.split(regex_pattern, input())))

# Group(), Groups() & Groupdict()

m = re.search(r"([a-zA-Z0-9])\1+", input())
print(m.group(1) if m else -1)

# Re.findall() & Re.finditer()

vowels = 'aeiouAEIOU'
a = re.findall(r'(?<=[^%s])([%s]{2,})[^%s]' %
               (vowels, vowels, vowels), input())
print('\n'.join(a or ['-1']))

# Re.start() & Re.end()

S = input()
k = input()
m = re.search(k, S)
if not m:
    print("(-1, -1)")
while m:
    print("({0}, {1})".format(m.start(), m.end()-1))
    m = re.compile(k).search(S, m.start()+1)

# Regex Substitution

reg = r'(?<= )(&&|\|\|)(?= )'

print('\n'.join(re.sub(reg, lambda x: 'and' if x.group() ==
      '&&' else 'or', input()) for _ in range(int(input().strip()))))

# Validating phone numbers


reg = r'[789][0-9]{9}$'
[print('YES' if re.match(reg, input()) else 'NO') for _ in range(int(input()))]

# Validating and Parsing Email Addresses

# reg for email validation... iso wasn't working?
regex = r'^[a-zA-Z]+[a-zA-Z0-9_.-]+[@][a-zA-Z]+[.][a-zA-Z]{1,3}$'

for _ in range(int(input())):
    a, b = map(str, email.utils.parseaddr(input()))
    if re.search(regex, b):
        print(email.utils.formataddr((a, b)))

# Hex Color Code

reg = r'(#[0-9a-fA-F]{3,6})[^\n ]'
for _ in range(int(input())):
    [print(x) for x in re.findall(reg, input())]

# HTML Parser - Part 1


class HTMLCustomParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print('Start :', tag)
        for el in attrs:
            print('->', el[0], '>', el[1])

    def handle_endtag(self, tag):
        print('End   :', tag)

    def handle_startendtag(self, tag, attrs):
        print('Empty :', tag)
        for el in attrs:
            print('->', el[0], '>', el[1])


MyParser = HTMLCustomParser()
MyParser.feed(''.join([input().strip() for _ in range(int(input()))]))

# HTML Parser - Part 2


class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):
        lines = len(data.split('\n'))
        if lines > 1:
            print('>>> Multi-line Comment')
        else:
            print('>>> Single-line Comment')
        if data.strip():
            print(data)

    def handle_data(self, data):
        if data.strip():
            print(">>> Data")
            print(data)


html = ""
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'

parser = MyHTMLParser()
parser.feed(html)
parser.close()


# Detect HTML Tags, Attributes and Attribute Values


class CustomHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        [print('-> {} > {}'.format(*attr)) for attr in attrs]


html = '\n'.join([input() for _ in range(int(input()))])
parser = CustomHTMLParser()
parser.feed(html)
parser.close()

# XML 1 - Find the Score


def get_attr_number(node):
    return (sum([len(el.attrib) for el in node.iter()]))


if __name__ == '__main__':
    sys.stdin.readline()
    xml = sys.stdin.read()
    tree = etree.ElementTree(etree.fromstring(xml))
    root = tree.getroot()
    print(get_attr_number(root))

# Validating UID


for _ in range(int(input())):
    s = ''.join(sorted(input()))

    checks = [r'[A-Z]{2,}', r'[0-9]{3,}', r'[^a-zA-Z0-9]{0}', r'.{10}']
    control = True
    i = 0
    while(control and i < len(checks)):
        control = re.search(checks[i], s) != None
        i += 1
    if(control):
        control = re.search(r'(.)\1', s) == None
    print('Valid') if control else print('Invalid')


# Validating Credit Card Numbers


reg1 = r'^[456](\d){15}(?!(.)\{5,})'
reg2 = r'(?=^[456])(?:-?\d{4}){4}'
reg3 = r'(\d)(-?\1){3}'

for _ in range(int(input().strip())):
    cc = input().strip()
    print("Valid" if (re.search(reg1, cc) or re.search(reg2, cc))
          and not re.search(reg3, cc) else "Invalid")

# Validating Credit Card Numbers


reg1 = r'^[456](\d){15}(?!(.)\{5,})$'
reg2 = r'(?=^[456])(?=(.){19})(?:-?\d{4}){4}$'
reg3 = r'(\d)(-?\1){3}'

for _ in range(int(input().strip())):

    cc = input().strip()
    print("Valid" if (re.search(reg1, cc) or re.search(reg2, cc))
          and not re.search(reg3, cc) else "Invalid")

# XML2 - Find the Maximum Depth

maxdepth = 0


def depth(elem, level):
    global maxdepth
    level += 1
    if level >= maxdepth:
        maxdepth = level
    for child in elem:
        depth(child, level)

# Standardize Mobile Number Using Decorators


if __name__ == '__main__':
    n = int(input())
    xml = ""
    for i in range(n):
        xml = xml + input() + "\n"
    tree = etree.ElementTree(etree.fromstring(xml))
    depth(tree.getroot(), -1)
    print(maxdepth)


def wrapper(f):
    def fun(l):
        decorated_l = ['+91 {} {}'.format(n[-10: -5], n[-5:]) for n in l]
        return f(decorated_l)
    return fun


@wrapper
def sort_phone(l):
    print(*sorted(l), sep='\n')


if __name__ == '__main__':
    l = [input() for _ in range(int(input()))]
    sort_phone(l)

# Decorators 2 - Name Directory


def person_lister(f):
    def inner(people):
        return map(f, sorted(people, key=lambda x: int(x[2])))
    return inner


@person_lister
def name_format(person):
    return ("Mr. " if person[3] == "M" else "Ms. ") + person[0] + " " + person[1]


if __name__ == '__main__':
    people = [input().split() for i in range(int(input()))]
    print(*name_format(people), sep='\n')


# Matrix Script

#!/bin/python3


first_multiple_input = input().rstrip().split()

n = int(first_multiple_input[0])

m = int(first_multiple_input[1])

matrix = []

for _ in range(n):
    matrix_item = input()
    matrix.append(matrix_item)

decoded = ''
for char in [*zip(*matrix)]:
    decoded += ''.join(char)
print(re.sub(r'\b[^a-zA-Z0-9]+\b', " ", decoded))
