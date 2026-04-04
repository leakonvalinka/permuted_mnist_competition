"""
Evaluate the softmax linear agent on the Permuted MNIST environment.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import time
import numpy as np
from permuted_mnist.env.permuted_mnist import PermutedMNISTEnv
from agent import Agent


def main():
    num_episodes = 10
    env = PermutedMNISTEnv(number_episodes=num_episodes)
    env.set_seed(42)


    total_time = 0
    accuracies = []

    for episode in range(num_episodes):

        task = env.get_next_task()
        if task is None:
            break

        agent = Agent()

        start = time.time()
        agent.train(task["X_train"], task["y_train"])
        predictions = agent.predict(task["X_test"])
        elapsed = time.time() - start

        if elapsed > 60:
            raise TimeoutError(f"One episode is timed out with {elapsed:0.02f}s")

        total_time += elapsed
        accuracy = env.evaluate(predictions, task["y_test"])
        accuracies.append(accuracy)

        print(f"Episode {episode + 1:2d}: Accuracy: {accuracy:.4f}, Time: {elapsed:.2f}s")

    print(f"\nMean Accuracy: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")
    print(f"Averaged Time:    {np.mean(total_time):.2f}s (+/- {np.std(total_time):.2f})")


if __name__ == "__main__":
    main()
