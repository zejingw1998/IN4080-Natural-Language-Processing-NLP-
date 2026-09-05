from typing import List
import re

def basic_tokenize(text: str) -> List[str]:
    """The method should split the text on white space, except for punctuation
    markers that should be considered as tokens of their own (even in the 
    absence of white space before or after their occurrence)"""

    # Implement here your basic tokenisation
    tokens = re.findall(r'\w+|[^\w\s]', text) 
    return tokens

text = "Pierre, who works at NR, also teaches at UiO."

tokens = basic_tokenize(text)

print(tokens)
print(len(tokens))

with open("ndt_test_lm.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokens = basic_tokenize(text)

print("Number of tokens:", len(tokens))
print("Number of types:", len(set(tokens)))

from typing import Dict, List, Tuple, Iterator
import numpy as np
from tqdm.notebook import tqdm

class BPETokenizer:
    """Tokenizer based on the Byte-Pair Encoding algorithm. 
    Note: the current implementation is limited to Latin characters (ISO-8859-1)"""

    def __init__(self, train_corpus_file: str, vocab_size = 5000):
        """Creates a new BPE tokenizer, with merge pairs found using the given
        corpus file. The extraction of merge pairs stops when a vocabulary of 
        size vocab_size is reached."""

        # List of string pairs that should be merged when tokenizing
        # Example: ('e', 't'), which means that 'et' is a possible subword
        # Each string pair is mapped to an unique index number
        # (corresponding to their position in the self.vocab list)
        self.merge_pairs = {}

        # We add as basic vocab all characters of the extended ASCII
        self.vocab = [chr(i) for i in range(256)]

        with open(train_corpus_file) as fd:

            # We first read the corpus, split on white space, and counts the
            # occurrences of each distinct word
            print("Counting word occurrences in corpus %s"%train_corpus_file, end="...", flush=True)
            text = fd.read()
            vocabulary_counts = {}
            for token in text.split():
                vocabulary_counts[token] = vocabulary_counts.get(token, 0) + 1
            print("Done")

            # We then iteratively extend the list of merge pairs until we
            # reach the desired size. Note: to speed up the algorithm, we 
            # extract n merge pairs at each iteration
            progress_bar = tqdm(total=vocab_size)
            while len(self.vocab) < vocab_size:
                most_common_pairs = self.get_most_common_pairs(vocabulary_counts)
                for common_pair in most_common_pairs:
                    self.merge_pairs[common_pair] = len(self.vocab)
                    self.vocab.append("".join(common_pair))
                progress_bar.update(len(most_common_pairs))
         #       print("Examples of new subwords:", ["".join(pair) for pair in most_common_pairs][:10])
            
    def get_most_common_pairs(self, vocabulary_counts: Dict[str,int], 
                              n:int=200) -> List[Tuple[str,str]]:
        """Given a set of distinct words along with their corresponding number 
        of occurrences in the corpus, returns the n most frequent pairs of subwords.       
        """

        # We count the frequencies of consecutive subwords in the vocabulary list
        pair_freqs = {}
        for word, word_count in vocabulary_counts.items():
            subwords = self.tokenize_word(word)
            for i in range(len(subwords)-1):
                byte_pair = (subwords[i], subwords[i+1])
                pair_freqs[byte_pair] = pair_freqs.get(byte_pair, 0) + word_count

        # And return the most frequent ones
        most_freq_pairs = sorted(pair_freqs.keys(), key=lambda x: pair_freqs[x])[::-1][:n]
        return most_freq_pairs

    def __call__(self, input:str, show_progress_bar=True) -> Iterator[str]:
        """Tokenizes a full text"""

        # We first split into whitespace-separated tokens, and then in subwords
        words = input.split()
        for word in tqdm(words) if show_progress_bar else words:
            subwords = self.tokenize_word(word)
            for subword in subwords:
                yield subword
                

    def tokenize_word(self, word):
        """Splits the word into subwords, according to the merge pairs 
        currently stored in self.merge_pairs."""

        # We start with a list of characters
        # (+ a final character to denote the end of the word)    
        splits = list(word) + [" "]

        # We continue until there is nothing left to be merged
        while len(splits)>=2:

            # We extract consecutive subword pairs
            pairs = [(splits[i], splits[i+1]) for i in range(len(splits)-1)]

            # We find the "best" pair of subwords to merge -- that is, the one with the 
            # lowest position in the list of merge rules
            best_pair_to_merge = min(pairs, key=lambda x: self.merge_pairs.get(x, np.inf))
            if best_pair_to_merge in self.merge_pairs:

                # We then merge the two subwords
                for i in range(len(splits)-1):
                    if (splits[i], splits[i+1]) == best_pair_to_merge:
                        merged_subword = self.vocab[self.merge_pairs[best_pair_to_merge]]
                        splits = splits[:i] + [merged_subword] + splits[i+2:]
                        break
            else:
                break
        return splits



#Learn the BPE tokenizer from the traing corpus

bpe_tokenizer = BPETokenizer("ndt_train_lm.txt", vocab_size=5000)


# Read the test corpus
with open("ndt_test_lm.txt") as fd:
    test_text = fd.read()


#Tokenize the test corpus

bpe_tokens = list(bpe_tokenizer(test_text))


#Count token and types

number_of_tokens = len(bpe_tokens)
number_of_types = len(set(bpe_tokens))


basic_tokens = basic_tokenize(test_text)

print("Basic tokenizer:")
print("Number of tokens:", len(basic_tokens))
print("Number of types:", len(set(basic_tokens)))

print()

print("BPE tokenizer:")
print("Number of tokens:", len(bpe_tokens))
print("Number of types:", len(set(bpe_tokens)))

#Conclusion

#The BPE tokenizer produces more tokens than the basic tokenizer, but fewer distinct types.
#This is beacause BPE splits words into smaller reusable subwords,
#Therefor one word may be represented by several BPE tokens
#While the vocabulary is much smaller.

#Improve the algorithm
from typing import Dict, List, Tuple, Iterator
import numpy as np
from tqdm.notebook import tqdm

class BPETokenizer:
    """Tokenizer based on the Byte-Pair Encoding algorithm. 
    Note: the current implementation is limited to Latin characters (ISO-8859-1)"""

    def __init__(self, train_corpus_file: str, vocab_size = 5000):
        """Creates a new BPE tokenizer, with merge pairs found using the given
        corpus file. The extraction of merge pairs stops when a vocabulary of 
        size vocab_size is reached."""

        # List of string pairs that should be merged when tokenizing
        # Example: ('e', 't'), which means that 'et' is a possible subword
        # Each string pair is mapped to an unique index number
        # (corresponding to their position in the self.vocab list)
        self.merge_pairs = {}

        # We add as basic vocab all characters of the extended ASCII
        self.vocab = [chr(i) for i in range(256)]

        with open(train_corpus_file) as fd:

            # We first read the corpus, split on white space, and counts the
            # occurrences of each distinct word
            print("Counting word occurrences in corpus %s"%train_corpus_file, end="...", flush=True)
            text = fd.read()
            vocabulary_counts = {}
            for token in basic_tokenize(text): #Changed here
                vocabulary_counts[token] = vocabulary_counts.get(token, 0) + 1
            print("Done")

            # We then iteratively extend the list of merge pairs until we
            # reach the desired size. Note: to speed up the algorithm, we 
            # extract n merge pairs at each iteration
            progress_bar = tqdm(total=vocab_size)
            while len(self.vocab) < vocab_size:
                most_common_pairs = self.get_most_common_pairs(vocabulary_counts)
                for common_pair in most_common_pairs:
                    self.merge_pairs[common_pair] = len(self.vocab)
                    self.vocab.append("".join(common_pair))
                progress_bar.update(len(most_common_pairs))
         #       print("Examples of new subwords:", ["".join(pair) for pair in most_common_pairs][:10])
            
    def get_most_common_pairs(self, vocabulary_counts: Dict[str,int], 
                              n:int=200) -> List[Tuple[str,str]]:
        """Given a set of distinct words along with their corresponding number 
        of occurrences in the corpus, returns the n most frequent pairs of subwords.       
        """

        # We count the frequencies of consecutive subwords in the vocabulary list
        pair_freqs = {}
        for word, word_count in vocabulary_counts.items():
            subwords = self.tokenize_word(word)
            for i in range(len(subwords)-1):
                byte_pair = (subwords[i], subwords[i+1])
                pair_freqs[byte_pair] = pair_freqs.get(byte_pair, 0) + word_count

        # And return the most frequent ones
        most_freq_pairs = sorted(pair_freqs.keys(), key=lambda x: pair_freqs[x])[::-1][:n]
        return most_freq_pairs

    def __call__(self, input:str, show_progress_bar=True) -> Iterator[str]:
        """Tokenizes a full text"""

        # We first split into whitespace-separated tokens, and then in subwords
        words = basic_tokenize(input) #Change here.
        for word in tqdm(words) if show_progress_bar else words:
            subwords = self.tokenize_word(word)
            for subword in subwords:
                yield subword
                

    def tokenize_word(self, word):
        """Splits the word into subwords, according to the merge pairs 
        currently stored in self.merge_pairs."""

        # We start with a list of characters
        # (+ a final character to denote the end of the word)    
        splits = list(word) + [" "]

        # We continue until there is nothing left to be merged
        while len(splits)>=2:

            # We extract consecutive subword pairs
            pairs = [(splits[i], splits[i+1]) for i in range(len(splits)-1)]

            # We find the "best" pair of subwords to merge -- that is, the one with the 
            # lowest position in the list of merge rules
            best_pair_to_merge = min(pairs, key=lambda x: self.merge_pairs.get(x, np.inf))
            if best_pair_to_merge in self.merge_pairs:

                # We then merge the two subwords
                for i in range(len(splits)-1):
                    if (splits[i], splits[i+1]) == best_pair_to_merge:
                        merged_subword = self.vocab[self.merge_pairs[best_pair_to_merge]]
                        splits = splits[:i] + [merged_subword] + splits[i+2:]
                        break
            else:
                break
        return splits



#Conclusion


#Changed for token in text.split() to for token in basic_tokenize(text)
#And words = text.split() to basic_tokenize(text)

#This separates punctuation from words before BPE is applied

# preventing punctuation markers from being merged with letters.



#Test the improved BPE tokenizer
bpe_tokenizer = BPETokenizer("ndt_train_lm.txt",vocab_size =5000)

test_text = "working,testing!"

tokens=list(bpe_tokenizer(test_text,show_progress_bar=False))


print(tokens)


# The punctuation marks are separated from the letter-based subwords,
# which shows that punctuation is no longer merged with letters.

import numpy as np
from abc import abstractmethod

class LanguageModel:
    """Generic class for running operations on language models, using a BPE tokenizer"""

    def __init__(self, tokenizer: BPETokenizer):
        """Build an abstract language model using the provided tokenizer"""
        self.tokenizer = tokenizer
 
    @abstractmethod
    def predict(self, context_tokens: List[str]):
        """Given a list of context tokens (=previous tokens), returns a dictionary
          mapping each possible token to its probability"""
        raise NotImplementedError()
    
    @abstractmethod
    def get_perplexity(self, text: str):
        """Computes the perplexity of the given text according to the LM"""

        print("Tokenising input text:")
        tokens = list(self.tokenizer(text))
        
        print("Computing perplexity:")
        log_probs = 0
        for i in tqdm(range(len(tokens))):
            context_tokens = ["<s>"] + tokens[:i]
            predict_distrib = self.predict(context_tokens)

            # We add the log-probabilities
            log_probs += np.log(predict_distrib[tokens[i]])
            
        perplexity = np.exp(-log_probs/len(tokens))
        return perplexity

class NGramLanguageModel(LanguageModel):
    """Representation of a N-gram-based language model"""

    def __init__(self, training_corpus_file: str, tokenizer:BPETokenizer, ngram_size:int=3,
                  alpha_smoothing:float=1):
        """Initialize the N-gram model with:
        - a file path to a training corpus to estimate the N-gram probabilities
        - an already learned BPE tokenizer
        - an N-gram size
        - a smoothing parameter (Laplace smoothing)"""
        
        LanguageModel.__init__(self, tokenizer)
        self.ngram_size = ngram_size
        
        # We define a simple backoff distribution (here just a uniform distribution)
        self.default_distrib = {token:1/len(tokenizer.vocab) for token in tokenizer.vocab}

        # Dictionary mapping a context (for instance the two preceding words if ngram_size=3)
        # to another dictionary specifying the probability of each possible word in the 
        # vocabulary. The context should be a tuple of tokens.
        self.ngram_probs = {}
        with open(training_corpus_file) as fd:   

            # based on the training corpus, tokenizer, ngram-size and smoothing parameter,
            # fill the self.ngram_probs with the correct N-gram probabilities  
            raise NotImplementedError()
 

    def predict(self, context_tokens: List[str]):
        """Given a list of preceding tokens, returns the probability distribution 
        over the next possible token."""

        # We restrict the contextual tokens to (N-1) tokens
        context_tokens = tuple(context_tokens[-self.ngram_size+1:])

        # If the contextual tokens were indeed observed in the corpus, simply
        # returns the precomputed probabilities
        if context_tokens in self.ngram_probs:
            return self.ngram_probs[context_tokens]
        
        # Otherwise, we return a uniform distribution over possible tokens
        else:
            return self.default_distrib



from collections import defaultdict, Counter

def __init__(self, training_corpus_file: str, tokenizer:BPETokenizer, ngram_size:int=2, alpha_smoothing=0.1):
        """Initialize the N-gram model with:
        - a file path to a training corpus to estimate the N-gram probabilities
        - an already learned BPE tokenizer
        - an N-gram size
        - a smoothing parameter (Laplace smoothing)"""
        LanguageModel.__init__(self, tokenizer)
        self.ngram_size = ngram_size
        
        # We define a simple backoff distribution (here just a uniform distribution)
        self.default_distrib = {token:1/len(tokenizer.vocab) for token in tokenizer.vocab}

        # Dictionary mapping a context (for instance the two preceding words if ngram_size=3)
        # to another dictionary specifying the probability of each possible word in the 
        # vocabulary. The context should be a tuple of tokens.
        self.ngram_probs = {}
        with open(training_corpus_file) as fd:  

            # ADD HERE YOUR CODE TO FILL THE VALUES IN self.ngram_probs

            text = fd.read()
            tokens =  list(self.tokenizer(text,show_progress_bar =False))
            
            counts = defaultdict(Counter)
            all_tokens = ["<s>"]+tokens 

            # Count the N-grams

            for i in range(1,len(all_tokens)):
                start = max(0, i - self.ngram_size + 1)
                
                context = tuple(all_tokens[start:i])

                next_token = all_tokens[i]
                counts[context][next_token]+=1 
            #Convert counts into probabilities
            for context, next_token_counts in counts.items():
                total_count = sum(next_token_counts.values())
                probabilities = {}
                
                for token in tokenizer.vocab:
                    probability = (next_token_counts[token] + alpha_smoothing) / (total_count + alpha_smoothing * len(tokenizer.vocab))
                    probabilities[token] = probability
                self.ngram_probs[context] = probabilities

setattr(NGramLanguageModel, '__init__', __init__)
#Test

ngram_model = NGramLanguageModel("ndt_train_lm.txt",bpe_tokenizer,ngram_size=2,alpha_smoothing=0.1)
print(len(ngram_model.ngram_probs))
context = next(iter(ngram_model.ngram_probs))

print("Context",context)
print("Sum of probabilities",sum(ngram_model.ngram_probs[context].values()))


#Since the sum of probabilities is 1.0, so this is correct. 


#Decompose sentence and label
sentences = []
labels = []

with open("ndt_train_class.txt", "r", encoding="utf-8") as f:

    for line in f:

        sentence, language = line.strip().split("\t")

        sentences.append(sentence)

        if language == "nno":  
            labels.append(1)
        else:
            labels.append(0)

#N x V matrix


N = len(sentences)
V = len(bpe_tokenizer.vocab)
X = np.zeros((N, V))
token_to_index = {token: i for i, token in enumerate(bpe_tokenizer.vocab)}
for i, sentence in enumerate(sentences):

    tokens = list(bpe_tokenizer(sentence, show_progress_bar=False))

    for token in tokens:

        if token in token_to_index: 
            j = token_to_index[token]

            X[i, j] = 1
y = np.array(labels)
print(X.shape)
print(y.shape)

#Conclusion

#This means the train text have 29905 sentences.

#And in the BPE vocabulary  have 5056 tokens


from sklearn.metrics import accuracy_score,recall_score,precision_score


test_sentences = []
test_labels = []

with open("ndt_test_class.txt","r",encoding="utf-8") as f:
    for line in f: 
        sentence,language = line.strip().split("\t")
        test_sentences.append(sentence)
        if language == "nno":
            test_labels.append(1)
        else:
            test_labels.append(0)


#Construct the test matrix

N_test = len(test_sentences)
V = len(bpe_tokenizer.vocab)

X_test = np.zeros((N_test,V))


for i,sentence in enumerate(test_sentences):
    tokens = list(bpe_tokenizer(sentence,show_progress_bar=False))
    for token in tokens:
        if token in tokens:
            if token in token_to_index:
                j = token_to_index[token ] 
                X_test[i,j] =1 

y_test = np.array(test_labels)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)

print("Accuracy",accuracy)
print("Recall",recall)
print("Precision",precision)


#Conclusion

#The model performs well with about 94.6% accuracy
#It also has high precision and recall for nynorsk


weight = model.coef_[0]

#Top 5 for nynorsk
nynorsk_indices = np.argsort(weight)[-5:][::-1]

print("To5 5 Nynorsk subwords:")

for i in nynorsk_indices:
    print(repr(bpe_tokenizer.vocab[i]),weight[i])

#Top 5 for bokmål

bokmål_indices = np.argsort(weight)[:5]


print("\n Top 5 Bokmål subwords:")
for i in bokmål_indices:
    print(repr(bpe_tokenizer.vocab[i]),weight[i])



