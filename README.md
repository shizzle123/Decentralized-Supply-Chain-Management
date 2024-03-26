# Decentralized-Supply-Chain-Management
利用区块链技术和机器学习算法，优化供应链管理，提高效率并降低成本。
import hashlib
import json
from time import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

# Part 1: Basic Blockchain for Supply Chain Traceability
class Blockchain:
    def __init__(self):
        self.chain = []
        self.pending_transactions = []
        self.new_block(previous_hash="1", proof=100)

    def new_block(self, proof, previous_hash=None):
        block = {
            'index': len(self.chain) + 1,
            'timestamp': time(),
            'transactions': self.pending_transactions,
            'proof': proof,
            'previous_hash': previous_hash or self.hash(self.chain[-1]),
        }
        self.pending_transactions = []
        self.chain.append(block)
        return block

    @staticmethod
    def hash(block):
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    def new_transaction(self, sender, recipient, product_id, quantity):
        self.pending_transactions.append({
            'sender': sender,
            'recipient': recipient,
            'product_id': product_id,
            'quantity': quantity,
        })

    @staticmethod
    def valid_chain(chain):
        # Implement validation logic (simplified here)
        return True

# Part 2: Machine Learning for Demand Forecasting
class DemandForecasting:
    def __init__(self, data):
        self.data = data

    def preprocess_data(self):
        # Preprocess your data here (simplified)
        X = self.data.drop('demand', axis=1)
        y = self.data['demand']
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self, X_train, y_train):
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    def predict_demand(self, model, X_test):
        return model.predict(X_test)

# Example usage
if __name__ == "__main__":
    # Initialize blockchain
    blockchain = Blockchain()
    blockchain.new_transaction("Manufacturer", "Distributor", "P123", 100)
    blockchain.new_transaction("Distributor", "Retailer", "P123", 90)
    block = blockchain.new_block(12345)

    print("Blockchain:", json.dumps(blockchain.chain, indent=2))

    # Demand forecasting with dummy data
    data = pd.DataFrame({
        'product_id': np.random.randint(1, 100, 100),
        'time_of_year': np.random.randint(1, 4, 100),
        'price': np.random.random(100) * 100,
        'demand': np.random.randint(1, 1000, 100),
    })

    forecasting = DemandForecasting(data)
    X_train, X_test, y_train, y_test = forecasting.preprocess_data()
    model = forecasting.train_model(X_train, y_train)
    predictions = forecasting.predict_demand(model, X_test)

    print("Demand predictions:", predictions[:5])
