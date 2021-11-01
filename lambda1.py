import os


def clear(): return os.system("cls")
# clear()


# def remainder(num):
# return num % 2

def remainder(num): return num % 2


print(type(remainder))
print(remainder(5))


def product(x, y): return x * y


print(product(2, 3))


def myfunction(num):
    return lambda x: x * num


result10 = myfunction(10)
result100 = myfunction(100)

print(result10(9))
print(result100(9))


def result10(x): return x * 10


def myfunc(n):
    return lambda a: a * n


mydoubler = myfunc(2)  # this is n
mytripler = myfunc(3)

print(mydoubler(11))  # this is a
print(mytripler(11))  # this is a

numbers = [2, 4, 68, 10, 3, 18, 14, 21]

filtered_list = list(filter(lambda num: (num > 7), numbers))
print(filtered_list)

mapped_list = list(map(lambda num: num % 2, numbers))
print(mapped_list)


def x(a): return a + 10


print(x(5))


def x(a, b, c): return a + b + c


print(x(5, 6, 7))


def addition(n):
    return n + n


numbers = [1, 2, 3, 4]
result = map(addition, numbers)
result = map(lambda x: x + x, numbers)
print(list(result))
