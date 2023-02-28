import tree
import naive

def main():
    print("****************************************************************************************")
    print("******** Neural network developed by Zachary Robinson, Hector Romero, Ian Hoole ********")
    print("****************************************************************************************")
    print("Please enter the two numbers we are testing for (q for either number to quit)")
    num1 = input("num1:")
    num2 = input("num2:")
    print("Creating data set for naive bayes(this will only happen once)")
    values = naive.train2()
    while num1 != "q" and num2 != "q":
        print("Naive Bayes success rate between", num1, "and", num2, ":", naive.naive(int(num1), int(num2), values))

        tree.main(num1, num2)

        print("Please enter the two numbers we are testing for (q for either number to quit)")
        num1 = input("num1:")
        num2 = input("num2:")
if __name__ == "__main__":
    main()