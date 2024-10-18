import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import lab functions
from lab2.lab2 import lab2
# from lab1.lab1 import lab1  # Uncomment when lab1 is ready
from lab3.lab3 import lab3  

def main():
    while True:
        print("\nComputer Vision Labs")
        print("1. Lab 1")
        print("2. Lab 2")
        print("3. Lab 3")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            print("Running Lab 1...")
            # lab1()  # Uncomment when lab1 is ready
        elif choice == '2':
            print("Running Lab 2...")
            lab2()
        elif choice == '3':
            print("Running Lab 3...")
            lab3()  
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()