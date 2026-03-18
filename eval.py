"""
Evaluate the softmax linear agent on the Permuted MNIST environment.
"""
import time
import numpy as np
from permuted_mnist.env.permuted_mnist import PermutedMNISTEnv
from agent import Agent


def main():
    num_episodes = 10
    env = PermutedMNISTEnv(number_episodes=num_episodes)
    env.set_seed(42)

    agent = Agent(input_dim=784, output_dim=10, learning_rate=0.05, epochs=5, batch_size=256)

    total_time = 0
    accuracies = []

    for episode in range(num_episodes):
        task = env.get_next_task()
        if task is None:
            break

        agent.reset()

        start = time.time()
        agent.train(task["X_train"], task["y_train"])
        predictions = agent.predict(task["X_test"])
        elapsed = time.time() - start

        total_time += elapsed
        accuracy = env.evaluate(predictions, task["y_test"])
        accuracies.append(accuracy)

        print(f"Episode {episode + 1:2d}: Accuracy: {accuracy:.4f}, Time: {elapsed:.2f}s")

    print(f"\nMean Accuracy: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")
    print(f"Total Time:    {total_time:.2f}s")
    print(f"Status:        {'PASS' if total_time < 600 else 'FAIL (timeout)'}")


if __name__ == "__main__":
    main()
