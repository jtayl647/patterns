def factorial(num, ans):
    if num < 0:
        print("Invalid number: factorial is not defined for negative numbers.")
        return 
    if num == 1:
        return ans

    ans *= num
    num -= 1
    return factorial(num, ans)

if __name__ == "__main__":
    try:
        num = int(input("Enter a number: "))
        print(factorial(num, 1))
    except ValueError:
        print("Please enter a valid integer.")