def fibonacci(num):
    if num < 0:
        print("Invalid Number")
    else:
        a, b = 0, 1
        i = 0
        while i <= num:
            print(a, end=" ")
            a, b = b, a + b
            i += 1
        print()

if __name__ == "__main__":

    num = int(input("Enter a number: "))
    fibonacci(num)