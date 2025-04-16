import torch
import numpy as np
import pandas as pd
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler, Dataset, random_split
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight

#Determining correct backend
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Training on Apple GPU")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Training on CUDA")
else:
    print ("MPS device not found.")

df = pd.read_csv("clean_COVIDSenti.csv")

reviews = df.iloc[:, 0].tolist()
encoded_labels = np.array([(label + 1) for label in df.iloc[:, 1].tolist()])
print(encoded_labels)
# Calculate class weights
negatives = 0
neutrals =0
positives = 0
for label in encoded_labels:
    if label == 0:
        negatives +=1
    elif label == 1:
        neutrals += 1
    else:
        positives += 1
print("there are ", negatives, "negatives")
print("there are ", neutrals, "neutrals")
print("there are ", positives, "positives")
print((negatives/len(encoded_labels))*100, "% neg \n")
print((neutrals/len(encoded_labels))*100, "% neutral \n")
print((positives/len(encoded_labels))*100, "% pos \n")

def pad_features(reviews_ints, seq_length):
    ''' Return features of review_ints, where each review is padded with 0's 
        or truncated to the input seq_length.
    '''
    ## getting the correct rows x cols shape
    features = np.zeros((len(reviews_ints), seq_length), dtype=int)
    
    ## for each review, I grab that review
    for i, row in enumerate(reviews_ints):
      features[i, -len(row):] = np.array(row)[:seq_length]
    
    return features

class TweetDataset(Dataset):
    def __init__(self, features, labels):
        self.x = features
        self.y = labels
        
    def __getitem__(self, index):
        #x = self.x[index]
        #y = self.y[index]
        #return (x, y)
        return torch.tensor(self.x[index], dtype=torch.long), int(self.y[index])
    
    def __len__(self):
        return len(self.x)


# First checking if GPU is available
train_on_gpu=torch.cuda.is_available()

if(train_on_gpu):
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')

class SentimentRNN(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super(SentimentRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True)
        
        # dropout layer
        self.dropout = nn.Dropout(0.3)
        
        # linear and sigmoid layer
        self.fc = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.Softmax()

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)
        
        # embeddings and lstm_out
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        
        # stack up lstm outputs
        lstm_out = lstm_out[:, -1, :]
        
        # dropout and fully connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        return out, hidden
    
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if(train_on_gpu):
          hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                   weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
          hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                   weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden


# dataloaders
batch_size = 50
folds = 5
train_frac = 0.8
test_frac = 0.1
val_frac = 0.1
test_accuracies = []
early_stopping = 5
#data = TweetDataset(features, encoded_labels)

# Five fold validation
k_folds = 5
skf = StratifiedKFold(n_splits = k_folds, shuffle = True, random_state=42)
labels = np.array(encoded_labels)


X = df
y = df['label']


for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    print(f"FOLD {fold}")

    train_indices_set = set(train_idx)
    test_indices_set = set(test_idx)
    assert len(train_indices_set.intersection(test_indices_set)) == 0, "Leakage: overlapping indices!"  

    #raw_tweets = df.iloc[:, 0].tolist()
    encoded_labels = np.array([(label + 1) for label in df.iloc[:, 1].tolist()])

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    train_fold_df, val_fold_df = train_test_split(
        train_df, test_size=0.1, stratify=train_df['label'], random_state=fold
    )

    train_labels = np.array([(label + 1) for label in train_fold_df.iloc[:, 1].tolist()])
    train_tweets = train_fold_df.iloc[:, 0].tolist()
    
    test_labels = np.array([(label + 1) for label in test_df.iloc[:, 1].tolist()])
    test_tweets = test_df.iloc[:, 0].tolist()

    val_labels = np.array([(label + 1) for label in val_fold_df.iloc[:, 1].tolist()])
    val_tweets = val_fold_df.iloc[:, 0].tolist()

    ## Build a dictionary that maps words to integers
    words = []
    for tweet in train_tweets:
        new_words = tweet.split()
        words.extend(new_words)

    counts = Counter(words)
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = {word: ii for ii, word in enumerate(vocab,1)}

    ## use the dict to tokenize each tweet in tweets_split
    ## store the tokenized tweets in tweets_ints
    train_tweets_ints = []
    for tweet in train_tweets:
        train_tweets_ints.append([vocab_to_int[word] for word in tweet.split()])

    #print(train_tweets_ints)

    seq_length = 28
    train_features = pad_features(train_tweets_ints, seq_length=seq_length)

    train_data = TweetDataset(train_features, train_labels)

    test_tweets_ints = []
    for tweet in test_tweets:
        full_tweet = []
        for word in tweet.split():
            if word in vocab_to_int:
                full_tweet.append(vocab_to_int[word])
            else: 
                full_tweet.append(0)
        test_tweets_ints.append(full_tweet)
    test_features = pad_features(test_tweets_ints, seq_length=seq_length)

    test_data = TweetDataset(test_features, test_labels)

    val_tweets_ints = []
    for tweet in val_tweets:
        full_tweet = []
        for word in tweet.split():
            if word in vocab_to_int:
                full_tweet.append(vocab_to_int[word])
            else: 
                full_tweet.append(0)
        val_tweets_ints.append(full_tweet)
    val_features = pad_features(val_tweets_ints, seq_length=seq_length)

    val_data = TweetDataset(val_features, val_labels)
    print(val_data)


    # Use Subset to create PyTorch datasets from indices
    #train_subset = torch.utils.data.Subset(data, train_idx)
    #test_subset = torch.utils.data.Subset(data, test_idx)

    # Optional: split a validation set from the training set
    #val_size = int(0.1 * len(train_data))
    #train_size = len(train_data) - val_size
    #train_dataset, val_dataset = random_split(train_data, [train_size, val_size])

    # Dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    # Instantiate the model w/ hyperparams
    vocab_size = len(vocab_to_int) + 1 # +1 for zero padding + our word tokens
    output_size = 3
    embedding_dim = 400 
    hidden_dim = 256
    n_layers = 2
    net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
    print(net)

    # loss and optimization functions
    lr=0.001
    #class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    #criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # training params

    epochs = 4 # 3-4 is approx where I noticed the validation loss stop decreasing
    epoch = 0
    counter = 0
    print_every = 100
    clip=5 # gradient clipping
    no_improvement = 0
    curr_acc = 0


    # move model to GPU, if available
    if(train_on_gpu):
        net.cuda()

    net.train()

    while no_improvement < early_stopping:
        epoch += 1
        print(f"Epoch {epoch}")

        # Training the model
        # initialize hidden state
        h = net.init_hidden(batch_size)

        # batch loop
        for inputs, labels in train_loader:
            counter += 1

            if(train_on_gpu):
                inputs, labels = inputs.cuda(), labels.cuda()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            net.zero_grad()

            # get the output from the model
            output, h = net(inputs, h)

            # calculate the loss and perform backprop
            loss = criterion(output, labels)
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()

            #Early stopping
            net.eval()
            correct = torch.tensor(0)
            incorrect = torch.tensor(0)

            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                all_preds = []
                all_labels = []
                for inputs, labels in val_loader:

                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])

                    if(train_on_gpu):
                        inputs, labels = inputs.cuda(), labels.cuda()

                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output, labels)

                    val_losses.append(val_loss.item())
                    preds = torch.argmax(output, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    correct += (preds == labels).sum()
                    incorrect += (preds != labels).sum()

                accuracy = correct / (correct + incorrect)
                if accuracy > curr_acc:
                    print(f"New accuracy has been reached: {accuracy}")
                    curr_acc = accuracy
                    no_improvement = 0
                else:
                    no_improvement += 1

                    

                net.train()
                print("Epoch: {}".format(epoch),
                    "Step: {}...".format(counter),
                    "Loss: {:.6f}...".format(loss.item()),
                    "Val Loss: {:.6f}".format(np.mean(val_losses)))
    
    net.eval()
    correct = torch.tensor(0)
    incorrect = torch.tensor(0)
    test_h = net.init_hidden(batch_size)
    test_losses = []
    all_preds = []
    all_labels = []
    
    #Getting test accuracy for CV purposes
    for inputs, labels in test_loader:
        test_h = tuple([each.data for each in test_h])
        if(train_on_gpu):
            inputs, labels = inputs.cuda(), labels.cuda()

        output, test_h = net(inputs, test_h)
        test_loss = criterion(output, labels)
        preds = torch.argmax(output, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        correct += (preds == labels).sum()
        incorrect += (preds != labels).sum()

    test_accuracy = correct / (correct + incorrect)
    test_accuracies.append(test_accuracy)
    print(f"FOR FOLD {fold}, THE TEST ACCURACY WAS {test_accuracy}")
    print("---------------------------------------")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    class_names = ["Negative", "Neutral", "Positive"]

    # Plotting
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Test Set)')
    plt.tight_layout()
    plt.show()

    # Precision-recall plotting 

    # Convert labels and preds to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    # Binarize the true labels for multi-class PR curve (One-vs-Rest style)
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])  # assuming 3 classes
    y_score_bin = label_binarize(y_pred, classes=[0, 1, 2])  # if you don't have probs, use predicted class for now

    # OPTIONAL: If you store raw `output` logits instead of argmax preds, you can use those for better curves
    # y_score_bin = np.vstack(all_logits)  # softmax or raw logits per class

    # Plot one PR curve per class
    class_names = ["Negative", "Neutral", "Positive"]
    plt.figure(figsize=(8, 6))
    for i in range(len(class_names)):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_score_bin[:, i])
        ap_score = average_precision_score(y_true_bin[:, i], y_score_bin[:, i])
        plt.plot(recall, precision, lw=2, label=f"{class_names[i]} (AP={ap_score:.2f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Test Set)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()