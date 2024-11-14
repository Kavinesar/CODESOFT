import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_text

data = pd.read_csv('F:\\codesoft\\task2\\transaction.csv')
X = data[['Amount', 'Location']]  
y = data['fstatus']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Classifier = DecisionTreeClassifier(max_depth=4, random_state=42)
Classifier.fit(X_train, y_train)
y_pred = Classifier.predict(X_test)
tree_rules = export_text(Classifier, feature_names=['Amount', 'Location'])
def fraudcheck(transaction_id):
    transaction = data[data['TransactionID'] == transaction_id]
    
    if transaction.empty:
        print(f"Transaction ID {transaction_id} not found.")
        return
    
    amount = transaction['Amount'].values[0]
    location = transaction['Location'].values[0]
    
    new_transaction = pd.DataFrame([[amount, location]], columns=['Amount', 'Location'])
    
    fraud_prediction = Classifier.predict(new_transaction)
    fraud_status = "Fraud" if fraud_prediction[0] == 1 else "Not Fraud"
    
    print(f"\nTransaction ID: {transaction_id}")
    print(f"Amount: Rs.{amount}")
    print(f"Fraud Status: {fraud_status}")

transaction_id = int(input("Enter Transaction ID to check: "))
fraudcheck(transaction_id)
