from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt

# Study hours
x = [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]

# Exam scores
y = [30, 40, 45, 55, 65, 70, 80, 85, 90, 95]

# Create the model
model = LinearRegression()

# Train the model
model.fit(x, y)

# Enter study hours
hours = float(input("Enter hours studied:"))

# Predict exam score
score = model.predict([[hours]])
print("Predicted Exam Score:", round(score[0], 2))

# Draw the graph
plt.scatter(x, y)
plt.plot(x,
         model.predict(x))
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Student ExamScore Prediction")
plt.show()
print("Graph displayed")

