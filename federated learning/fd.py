import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt

df = pd.read_csv("cicids.csv")

df['Label'] = df['Label'].apply(
    lambda x: 0 if x == 'BENIGN' else 1
)

X = df.drop(columns=['Label']).values
y = df['Label'].values

X = np.nan_to_num(X)

scaler = StandardScaler()

X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

NUM_CLIENTS = 2

X_clients = np.array_split(X_train, NUM_CLIENTS)
y_clients = np.array_split(y_train, NUM_CLIENTS)

def create_model():

    model = keras.Sequential([

        layers.Input(shape=(X_train.shape[1],)),

        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),

        layers.Dense(32, activation='relu'),

        layers.Dense(1, activation='sigmoid')

    ])

    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

def dynamic_svd_compression(
    weights,
    threshold=0.90
):

    compressed_weights = []

    for w in weights:

        if len(w.shape) == 2:

            U, S, Vt = np.linalg.svd(
                w,
                full_matrices=False
            )

            total_variance = np.sum(S ** 2)

            cumulative_variance = 0
            rank = 0

            for i in range(len(S)):

                cumulative_variance += S[i] ** 2
                rank += 1

                if cumulative_variance / total_variance >= threshold:
                    break

            U_r = U[:, :rank]
            S_r = np.diag(S[:rank])
            Vt_r = Vt[:rank, :]

            compressed = np.dot(
                U_r,
                np.dot(S_r, Vt_r)
            )

            compressed_weights.append(compressed)

        else:

            compressed_weights.append(w)

    return compressed_weights

ROUNDS = 5
LOCAL_EPOCHS = 2

global_model = create_model()

accuracy_history = []

print("\n======================================")
print("FEDERATED TRAINING STARTED")
print("======================================")

for round_num in range(ROUNDS):

    print(f"\nROUND {round_num+1}")

    local_weights = []

    for client_id in range(NUM_CLIENTS):

        local_model = create_model()

        local_model.set_weights(
            global_model.get_weights()
        )

        local_model.fit(
            X_clients[client_id],
            y_clients[client_id],
            epochs=LOCAL_EPOCHS,
            batch_size=32,
            verbose=0
        )

        client_pred = local_model.predict(
            X_clients[client_id],
            verbose=0
        )

        client_pred = (
            client_pred > 0.5
        ).astype(int)

        client_acc = accuracy_score(
            y_clients[client_id],
            client_pred
        )

        print(
            f"Client {client_id+1} Accuracy: "
            f"{client_acc:.4f}"
        )

        compressed_weights = dynamic_svd_compression(
            local_model.get_weights(),
            threshold=0.90
        )

        local_weights.append(compressed_weights)

    averaged_weights = []

    for weights_tuple in zip(*local_weights):

        averaged_weights.append(
            np.mean(
                weights_tuple,
                axis=0
            )
        )

    global_model.set_weights(
        averaged_weights
    )

    predictions = global_model.predict(
        X_test,
        verbose=0
    )

    predictions = (
        predictions > 0.5
    ).astype(int)

    global_acc = accuracy_score(
        y_test,
        predictions
    )

    accuracy_history.append(global_acc)

    print(
        f"Global Accuracy: "
        f"{global_acc:.4f}"
    )

print("\n======================================")
print("FINAL EVALUATION")
print("======================================")

print(
    f"\nFinal Accuracy: "
    f"{accuracy_history[-1]:.4f}"
)

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        predictions
    )
)

cm = confusion_matrix(
    y_test,
    predictions
)

print("Confusion Matrix:\n")

print(cm)

plt.figure(figsize=(8,5))

plt.plot(
    range(1, ROUNDS + 1),
    accuracy_history,
    marker='o'
)

plt.xlabel("Federated Rounds")
plt.ylabel("Accuracy")
plt.title("FedTDLR Accuracy")

plt.grid(True)

plt.show()