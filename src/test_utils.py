import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def evaluate_sample(preds, labels, threshold):
    """
    :param preds: predicted labels
    :param labels: true labels
    :param threshold: majority voting threshold (between 0 to 1)
    :return: correctness of classification after majority voting
    """

    matched = float(torch.sum(preds == labels))
    if matched / len(preds) >= threshold:
        return True
    return False


def evaluate(net, X, Y, verbose=False, threshold=0.6, n=200):
    """
    :param net: trained instance of our deep learning model
    :param X: dataset features for evaluation (tensor)
    :param Y: true labels for evaluation (tensor)
    :param verbose: Prints results in detail for every music recording
    :param threshold: majority voting threshold
    :param n: number of subsequences per music recording
    :return: accuracy over given dataset
    """

    net.eval()
    N = len(Y) // n  # no of music samples in test set
    correct = 0
    for i in range(N):
        start = i * n
        with torch.no_grad():
            out = net.forward(X[start:start + n].to(device))
        preds = torch.argmax(out, axis=-1)
        labels = Y[start:start + n].to(device)
        if evaluate_sample(preds, labels, threshold):
            correct += 1
            if verbose:
                print(f"Sample {i + 1}/{N} classified as CORRECT")
        elif verbose:
            print(f"Sample {i + 1}/{N} classified as INCORRECT")

    accuracy = correct / N
    if verbose:
        print("=" * 50)
        print(f"Accuracy of the model on {N} unseen music samples: {(accuracy * 100):.2f}%")
        print("=" * 50)
    return accuracy


def evaluate_naive(net, X, Y, n=200):
    """Evaluate the model performance independently on the
    subsequences in the dataset."""
    net.eval()
    N = len(Y) // n  # no of music samples in test set
    correct = 0
    for i in range(N):
        start = i * n
        with torch.no_grad():
            out = net.forward(X[start:start + n].to(device))
        preds = torch.argmax(out, axis=-1)
        labels = Y[start:start + n].to(device)
        correct += torch.sum(preds == labels)

    accuracy = float(correct) / len(Y)
    print("=" * 50)
    print(f"Accuracy on given data: {(accuracy * 100):.2f}%")
    print("=" * 50)
    return accuracy

mapping10 = {0: 'Suraṭi', 1: 'Mukhāri', 2: 'Varāḷi', 3: 'Ānandabhairavi', 4: 'Hussēnī',
             5: 'Aṭāna', 6: 'Madhyamāvati', 7: 'Dēvagāndhāri', 8: 'Kāṁbhōji', 9: 'Bēgaḍa'}

def predict10(net, X, threshold = 0.6, mapping=mapping10):
    """
    :param net: trained instance of our deep learning model
    :param X: all subsequences for a certain music recording
    :param threshold: majority voting threshold
    :param mapping: dictionary mapping labels to raga names
    :return: recognition result
    """

    net.eval()
    with torch.no_grad():
        out = net.forward(X.to(device))
    preds = torch.argmax(out, axis=-1)
    majority, _ = torch.mode(preds)
    majority = int(majority)
    votes = float(torch.sum(preds==majority))/X.shape[0]
    if votes >= threshold:
        return f"Input music sample belongs to the {mapping[majority]} raga"
    return f"CONFUSED - Closest raga predicted is {mapping[majority]} with {(votes*100):.2f}% votes"