import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import lab functions
from lab2.lab2 import lab2
from lab3.lab3 import lab3  
from lab4.lab4 import lab4
from lab5.lab5 import lab5
def main():
    while True:
        print("\nComputer Vision Labs")
        print("1. Exit")
        print("2. Lab 2")
        print("3. Lab 3")
        print("4. Lab 4")
        print("5. Lab 5")
        
        choice = input("Enter your choice (1-4): ")
        
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
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()