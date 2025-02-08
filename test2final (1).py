import scapy.all as scapy
import matplotlib.pyplot as plt
import numpy as np
import time
import threading

# Global variables to store statistics
packet_count = 0
correct_predictions = 0
total_predictions = 0

# Simulated accuracy and loss
def simulate_ml_training(num_iterations):
    global correct_predictions, total_predictions
    accuracy = []
    loss = []

    for _ in range(num_iterations):
        # Simulate predictions
        total_predictions += 1
        if np.random.rand() > 0.2:  # Simulate 80% accuracy
            correct_predictions += 1

        # Calculate accuracy and loss
        current_accuracy = correct_predictions / total_predictions
        current_loss = 1 - current_accuracy  # Simple mock loss function

        accuracy.append(current_accuracy)
        loss.append(current_loss)

        # Simulate time delay for realism
        time.sleep(0.5)

    return accuracy, loss

def packet_handler(packet):
    global packet_count
    packet_count += 1
    # Print packet summary (or analyze further)
    print(f"Packet captured: {packet.summary()}")

def start_monitoring(interface):
    print(f"Starting packet capture on {interface}...")
    scapy.sniff(iface=None, prn=packet_handler)


def plot_results(accuracy, loss):
    plt.figure(figsize=(12, 6))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(accuracy, label='Accuracy', color='blue', marker='o')
    plt.title('Model Accuracy Over Time')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.xticks(range(len(accuracy)))
    plt.grid()
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Loss', color='red', marker='x')
    plt.title('Model Loss Over Time')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.ylim(0, 1)
    plt.xticks(range(len(loss)))
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    interface = "ethernet2"  # Updated interface name
    num_iterations = 20

    try:
        monitoring_thread = threading.Thread(target=start_monitoring, args=(interface,))
        monitoring_thread.start()

        # Simulate ML training and collect results
        accuracy, loss = simulate_ml_training(num_iterations)

        # Wait for monitoring to complete (optional, can run indefinitely)
        monitoring_thread.join(timeout=10)

        # Plot results
        plot_results(accuracy, loss)

    except KeyboardInterrupt:
        print("Monitoring stopped by user.")
# compiles the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# trains the model
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val))
# evaluates the model on the test set
y_prediction = model.predict(X_test)
y_prediction = (y_prediction > 0.5) # Apply threshold (adjust as needed)
# creates a classification report
report = classification_report(y_test, y_prediction, target_names=['Normal', 'Anomaly'], zero_division=1)
# creating a loss variable from the trained model
loss = history.history['loss']
# creating a validation loss variable from the trained model
validation_loss = history.history['val_loss']
# creating an accuracy variable from the trained model
accuracy = history.history['accuracy']
# creating a validation accuracy variable from the trained model
validation_accuracy = history.history['val_accuracy']
# creating an epochs variable to show the number of samples used
epochs = range(1, len(loss) + 1)
# setting the plot graph size
plt.figure(figsize=(12, 5))
# creating a subplot to display the training loss over epochs on the left of the two

plt.subplot(1, 2, 1)
# setting the epoch and loss variables for the plot over epochs
plt.plot(epochs, loss, 'bo-', label='Training Loss')
# setting the epoch and validation loss variables for the plot over epochs
plt.plot(epochs, validation_loss, 'ro-', label='Validation Loss')
# the title of the subplot
plt.title('Loss over Epochs')
# labeling the x-axis
plt.xlabel('Epochs')
# labeling the y-axis
plt.ylabel('Loss')
# creating a legend for the subplot
plt.legend()
# creating a subplot to display the training accuracy over epochs on the right of the two
plt.subplot(1, 2, 2)
# setting the epoch and accuracy variables for the plot over epochs
plt.plot(epochs, accuracy, 'bo-', label='Training Accuracy')
# setting the epoch and validation accuracy variables for the plot over epochs
plt.plot(epochs, validation_accuracy, 'ro-', label='Validation Accuracy')
# the title of the subplot
plt.title('Accuracy over Epochs')
# labeling the x-axis
plt.xlabel('Epochs')
# labeling the y-axis
plt.ylabel('Accuracy')
# setting the y-axis to use regular decimal form
plt.gca().yaxis.set_major_formatter(ScalarFormatter())
# creating a legend for the subplot
plt.legend()
# prints the classification report
print(report)
