import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import lab functions
from lab2.lab2 import lab2
from lab3.lab3 import lab3  
from lab4.lab4 import lab4
from lab5.lab5 import lab5
from lab6.lab6 import lab6
from lab7.lab7 import lab7
def main():
    while True:
        print("\nComputer Vision Labs")
        print("1. Exit")
        print("2. Lab 2")
        print("3. Lab 3")
        print("4. Lab 4")
        print("5. Lab 5")
        print("6. Lab 6")
        print("7. Lab 7")
        print("8. Lab 8")
        print("9. Lab 9")
        print("10. Lab 10")
        print("11. Lab 11")
        
        choice = input("Enter your choice (1-11): ")
        
        if choice == '1':
            print("Exiting...")
            break
        elif choice == '2':
            print("Running Lab 2...")
            lab2()
        elif choice == '3':
            print("Running Lab 3...")
            lab3()  
        elif choice == '4':
            print("Running Lab 4...")
            lab4()  
        elif choice == '5':
            print("Running Lab 5...")
            lab5()  
        elif choice == '6':
            print("Running Lab 6...")
            lab6() 
        elif choice == '7':
            print("Running Lab 7...")
            lab7() 
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()