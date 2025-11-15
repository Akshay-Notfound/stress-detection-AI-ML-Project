"""
Simple run script to start the stress detection demo.
"""

import subprocess
import sys
import os

def main():
    print("Stress Detection System - Demo Runner")
    print("=" * 40)
    print()
    
    # Check if we're in the right directory
    if not os.path.exists("app") or not os.path.exists("src"):
        print("Error: Please run this script from the project root directory.")
        print("The directory should contain 'app' and 'src' folders.")
        return 1
    
    print("Available options:")
    print("1. Run Streamlit demo application")
    print("2. Run test pipeline")
    print("3. Start Jupyter notebooks")
    print("4. Exit")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (1-4): ").strip()
            
            if choice == "1":
                print("Starting Streamlit demo application...")
                try:
                    subprocess.run([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"], check=True)
                except subprocess.CalledProcessError:
                    print("Error: Failed to start Streamlit app. Make sure Streamlit is installed.")
                    print("Run 'pip install streamlit' to install it.")
                except FileNotFoundError:
                    print("Error: Streamlit not found. Please install dependencies first.")
                break
                
            elif choice == "2":
                print("Running test pipeline...")
                try:
                    subprocess.run([sys.executable, "test_pipeline.py"], check=True)
                except subprocess.CalledProcessError:
                    print("Error: Test pipeline failed.")
                except FileNotFoundError:
                    print("Error: test_pipeline.py not found.")
                break
                
            elif choice == "3":
                print("Starting Jupyter notebooks...")
                try:
                    subprocess.run([sys.executable, "-m", "jupyter", "notebook"], check=True)
                except subprocess.CalledProcessError:
                    print("Error: Failed to start Jupyter. Make sure Jupyter is installed.")
                    print("Run 'pip install jupyter' to install it.")
                except FileNotFoundError:
                    print("Error: Jupyter not found. Please install dependencies first.")
                break
                
            elif choice == "4":
                print("Goodbye!")
                break
                
            else:
                print("Invalid choice. Please enter a number between 1 and 4.")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            break

if __name__ == "__main__":
    main()