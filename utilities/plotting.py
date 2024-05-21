import matplotlib.pyplot as plt


def plot_metrics(train_losses, train_accuracies, test_accuracies):
    """
    Plot training loss, training accuracy, and test accuracy over epochs.

    Args:
        train_losses (list of float): List of training losses over epochs.
        train_accuracies (list of float): List of training accuracies over epochs.
        test_accuracies (list of float): List of test accuracies over epochs.
    """
    
    # Determine the number of epochs 
    epochs = range(1, len(train_losses) + 1)

    # Create a plot for training loss 
    fig, ax1 = plt.subplots(dpi=300)
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss', color=color)
    ax1.plot(epochs, train_losses, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a secondary axis to plot training accuracy
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Train Accuracy', color=color)  
    ax2.plot(epochs, train_accuracies, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    fig.savefig('train_metrics.png', bbox_inches='tight')
    print('\nPlot of training loss and training accuracy saved to train_metrics.png')

    # Plot testing accuracy 
    fig, ax = plt.subplots(dpi=300)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy')
    ax.plot(epochs, test_accuracies, label='Test Accuracy')
    ax.legend()

    fig.tight_layout()
    fig.savefig('test_accuracy.png', bbox_inches='tight')
    print('Plot of test accuracy saved to test_accuracy.png')


def log_metrics(epoch, train_loss, test_loss, train_acc, test_acc, best_acc, duration, learning_rate):
    """
    Log training and testing metrics to the console.

    Args:
        epoch (int): Current epoch number.
        train_loss (float): Training loss for the epoch.
        test_loss (float): Testing loss for the epoch.
        train_acc (float): Training accuracy for the epoch.
        test_acc (float): Testing accuracy for the epoch.
        best_acc (float): Best observed testing accuracy.
        duration (float): Duration of the epoch in seconds.
        learning_rate (float): Current learning rate.
    """
    print(f'Epoch: {epoch + 1}',
          f'Train Loss: {train_loss:.4f}',
          f'Test Loss: {test_loss:.4f}',
          f'Train Acc: {train_acc:.4f}',
          f'Test Acc: {test_acc:.4f}',
          f'Best Acc: {best_acc:.4f}',
          f'Time: {duration:.2f}s',
          f'LR: {learning_rate:.6f}', sep='  |  ')
