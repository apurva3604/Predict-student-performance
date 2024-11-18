import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Function to predict student performance based on study hours per day and attendance
def predict_performance(study_hours_per_day, attendance):
    if study_hours_per_day >= 1.5 and attendance >= 75:
        return "Pass"
    elif study_hours_per_day >= 1 and attendance >= 60:
        return "Pass"
    elif study_hours_per_day < 1 and attendance < 60:
        return "Fail"
    else:
        return "Uncertain"

# Function to calculate average study hours and attendance
def calculate_averages(data):
    avg_study_hours = np.mean(data[:, 0])
    avg_attendance = np.mean(data[:, 1])
    print(f"\nAverage Study Hours Per Day: {avg_study_hours:.2f}")
    print(f"Average Attendance: {avg_attendance:.2f}%")

# Function to visualize data with linear regression line
def visualize_data_with_regression(data, performance, model):
    plt.figure(figsize=(10, 6))

    # Define custom colors for Pass, Fail, and Uncertain
    color_map = {'Pass': 'darkgreen', 'Fail': 'darkred', 'Uncertain': 'darkgoldenrod'}

    # Create a scatter plot with the custom colors
    sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=performance, palette=color_map, s=100)

    # Prepare data for regression line
    x_values = np.linspace(min(data[:, 0]), max(data[:, 0]), 100).reshape(-1, 1)
    y_values = model.predict(x_values)  # Use only study hours for prediction

    # Plot the regression line
    plt.plot(x_values, y_values, color='blue', label='Regression Line', linewidth=2)

    # Plot a horizontal line for minimum attendance criteria (50%)
    plt.axhline(y=50, color='blue', linestyle='--', label='Minimum Attendance (50%)')

    plt.title('Study Hours vs Attendance (Performance Prediction with Regression)')
    plt.xlabel('Study Hours Per Day')
    plt.ylabel('Attendance (%)')
    plt.legend(title='Performance', loc='upper left')
    plt.show()
    plt.close()  # Close the figure

# Main program
def main():
    print("Student Performance Prediction using Supervised Learning")

    # Get user input for the number of students
    try:
        num_students = int(input("Enter the number of students: "))
    except ValueError:
        print("Invalid input! Please enter a valid number of students.")
        return

    # Initialize arrays to store data
    study_hours_list = []
    attendance_list = []

    # Collect study hours and attendance data for each student
    for i in range(num_students):
        try:
            study_hours = float(input(f"\nEnter study hours per day for Student {i+1}: "))
            attendance = float(input(f"Enter attendance percentage for Student {i+1}: "))

            # Check if inputs are valid
            if study_hours < 0 or attendance < 0 or attendance > 100:
                print("Invalid input! Study hours should be non-negative and attendance between 0 and 100.")
                return
        except ValueError:
            print("Please enter valid numeric values!")
            return

        study_hours_list.append(study_hours)
        attendance_list.append(attendance)

    # Convert input lists to numpy array for further processing
    data = np.array(list(zip(study_hours_list, attendance_list)))

    # Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(data[:, 0].reshape(-1, 1), data[:, 1], test_size=0.2, random_state=42)

    # Create and train the linear regression model (Supervised Learning)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict attendance for the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    print(f"\nMean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.2f}")
    print(f"R-squared (R2 Score): {r2_score(y_test, y_pred):.2f}")

    # Predict performance for each student
    performance = [predict_performance(study_hours, attendance) for study_hours, attendance in data]

    # Print average study hours and attendance
    calculate_averages(data)

    # Visualize the data with regression line
    visualize_data_with_regression(data, performance, model)

    # Display predictions for each student
    for i in range(num_students):
        print(f"\nStudent {i+1}: Study Hours = {data[i][0]:.2f}, Attendance = {data[i][1]:.2f}% -> Prediction: {performance[i]}")

# Run the program
if __name__ == "__main__":
    main()
