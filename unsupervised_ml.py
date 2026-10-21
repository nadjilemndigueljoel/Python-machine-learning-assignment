from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Customer data
customers = [
    [1000, 2],
    [1500, 3],
    [2000, 2],
    [5000, 6],
    [6000, 7],
    [5500, 5],
    [12000, 12],
    [15000, 14],
    [13000, 11],
    [3000, 4]
]

# Scale the data
scaler = StandardScaler()
data = scaler.fit_transform(customers)

# Create the K-Means()
model = KMeans(n_clusters=3,
               random_state=42, n_init=10)

# Train the model
model.fit(data)

# Display the cluster of each customer
for i in range(len(customers)):
        print("customer", i + 1, 
      "Cluster:",model.labels_[i])

# Display the number ofcustomers in each cluster        
print("\number of customers in each cluster:")

for i in range(3):
        print("Cluster", i, ".",
              list(model.labels_).
              count(i))

# Draw the customer groups
plt.scatter(
        [customer[0] for
     customer in customers],
         [customer [1] for
     customer in customers],
          c = model.labels_
)

plt.xlabel("Monthly Spending(k)")
plt.ylabel("Monthly Visits")
plt.title("Customer Segmentation")
plt.show()
