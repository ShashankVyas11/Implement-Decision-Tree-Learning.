from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree # Import plot_tree
from sklearn.metrics import accuracy_score, classification_report,
confusion_matrix
import matplotlib.pyplot as plt
# Load a dataset (e.g., the Iris dataset)
data = datasets.load_iris()
X = data.data
y = data.target
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
random_state=42)
# Create a Decision Tree Classifier
clf = DecisionTreeClassifier()
# Fit the classifier to the training data
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
# Calculate accuracy

accuracy = accuracy_score(y_test, y_pred)
print(f&quot;Accuracy: {accuracy:.2f}&quot;)
# Generate a classification report
report = classification_report(y_test, y_pred)
print(&quot;Classification Report:&quot;)
print(report)
# Generate a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(&quot;Confusion Matrix:&quot;)
print(conf_matrix)
# Visualize the Decision Tree using plot_tree
plt.figure(figsize=(15, 10))
plot_tree(clf, filled=True) # Use plot_tree from sklearn.tree
plt.show()