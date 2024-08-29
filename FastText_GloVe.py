import nltk
import spacy
import stanza
from gensim.models import Word2Vec, FastText
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time
import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


dataset = load_dataset('universal_dependencies', 'en_ewt', trust_remote_code=True)
print("Пример данных из train датасета:", dataset['train'][0])

def prepare_data(data):
    sentences, labels = [], []
    for item in data:
        sentences.append(item['tokens'])
        labels.append(item['upos'])
    return sentences, labels

train_data = dataset['train']
test_data = dataset['test']

train_sentences, train_labels = prepare_data(train_data)
test_sentences, test_labels = prepare_data(test_data)

upos_tags = {
    0: 'ADJ', 1: 'ADP', 2: 'ADV', 3: 'AUX', 4: 'CCONJ', 5: 'DET', 6: 'INTJ', 
    7: 'NOUN', 8: 'NUM', 9: 'PART', 10: 'PRON', 11: 'PROPN', 12: 'PUNCT', 
    13: 'SCONJ', 14: 'SYM', 15: 'VERB', 16: 'X'
}

def train_word2vec(sentences, sg):
    model = Word2Vec(vector_size=100, window=5, sg=sg, min_count=1, workers=4)
    model.build_vocab(corpus_iterable=sentences)
    model.train(corpus_iterable=sentences, total_examples=len(sentences), epochs=10)
    return model

w2v_skipgram_model = train_word2vec(train_sentences, sg=1)
w2v_cbow_model = train_word2vec(train_sentences, sg=0)

def train_fasttext(sentences):
    model = FastText(vector_size=100, window=5, min_count=1, workers=4)
    model.build_vocab(corpus_iterable=sentences)
    model.train(corpus_iterable=sentences, total_examples=len(sentences), epochs=10)
    return model

fasttext_model = train_fasttext(train_sentences)

def generate_token_embeddings(model, sentences):
    token_embeddings = []
    for sent in sentences:
        word_vectors = [model.wv[word] for word in sent if word in model.wv]
        if not word_vectors:
            word_vectors = [np.zeros(model.vector_size) for _ in sent]
        token_embeddings.extend(word_vectors)
    return np.array(token_embeddings)

train_embeddings_w2v_skipgram = generate_token_embeddings(w2v_skipgram_model, train_sentences)
train_embeddings_w2v_cbow = generate_token_embeddings(w2v_cbow_model, train_sentences)
train_embeddings_fasttext = generate_token_embeddings(fasttext_model, train_sentences)

test_embeddings_w2v_skipgram = generate_token_embeddings(w2v_skipgram_model, test_sentences)
test_embeddings_w2v_cbow = generate_token_embeddings(w2v_cbow_model, test_sentences)
test_embeddings_fasttext = generate_token_embeddings(fasttext_model, test_sentences)

print(f"Размер train_embeddings_w2v_skipgram: {len(train_embeddings_w2v_skipgram)}")
print(f"Размер train_labels: {len(train_labels)}")
print(f"Размер test_embeddings_w2v_skipgram: {len(test_embeddings_w2v_skipgram)}")
print(f"Размер test_labels: {len(test_labels)}")

train_labels_flat = [upos_tags[tag] for sent in train_labels for tag in sent if tag in upos_tags]
test_labels_flat = [upos_tags[tag] for sent in test_labels for tag in sent if tag in upos_tags]


print(f"Размер train_labels_flat: {len(train_labels_flat)}")
print(f"Размер test_labels_flat: {len(test_labels_flat)}")
print(train_labels_flat[:10])
print(train_embeddings_w2v_skipgram.shape)
# Проверяем уникальные значения
unique_labels = np.unique(train_labels_flat)
print("Unique labels in train:", unique_labels)

if len(unique_labels) > 1:
    print("Data seems okay. Proceeding with model training.")
else:
    print("Data issue: Only one unique class present in training labels.")

def train_evaluate_logistic_regression(train_embeddings, test_embeddings, train_labels_flat, test_labels):
    X_train, X_val, y_train, y_val = train_test_split(train_embeddings, train_labels_flat, test_size=0.2, random_state=42)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(test_embeddings)
    
    accuracy = accuracy_score(test_labels, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, y_pred, average='macro')
    
    return accuracy, precision, recall, f1

print("\nОценка моделей на основе Word2Vec и FastText")

print(f"Shape of train_embeddings_w2v_skipgram: {np.array(train_embeddings_w2v_skipgram).shape}")
print(f"Shape of test_embeddings_w2v_skipgram: {np.array(test_embeddings_w2v_skipgram).shape}")

start_time = time.time()
acc_w2v_skipgram, prec_w2v_skipgram, rec_w2v_skipgram, f1_w2v_skipgram = train_evaluate_logistic_regression(
    train_embeddings_w2v_skipgram, test_embeddings_w2v_skipgram, train_labels_flat, test_labels_flat)
time_w2v_skipgram = time.time() - start_time

start_time = time.time()
acc_w2v_cbow, prec_w2v_cbow, rec_w2v_cbow, f1_w2v_cbow = train_evaluate_logistic_regression(
    train_embeddings_w2v_cbow, test_embeddings_w2v_cbow, train_labels_flat, test_labels_flat)
time_w2v_cbow = time.time() - start_time

start_time = time.time()
acc_fasttext, prec_fasttext, rec_fasttext, f1_fasttext = train_evaluate_logistic_regression(
    train_embeddings_fasttext, test_embeddings_fasttext, train_labels_flat, test_labels_flat)
time_fasttext = time.time() - start_time

# --- NLTK POS-теггинг ---
print("\nNLTK POS-теггинг")

start_time = time.time()

nltk_results = []
for sent in test_sentences:
    tags = nltk.pos_tag(sent, tagset='universal')
    nltk_results.append([upos_tags.get(tag, 17) for _, tag in tags])

nltk_time = time.time() - start_time

nltk_true_labels = [tag for sent in test_labels for tag in sent]
nltk_predicted_labels = [tag for sent in nltk_results for tag in sent]

nltk_accuracy = accuracy_score(nltk_true_labels, nltk_predicted_labels)
nltk_precision, nltk_recall, nltk_f1, _ = precision_recall_fscore_support(nltk_true_labels, nltk_predicted_labels, average='macro')

# --- SpaCy POS-теггинг ---
print("\nSpaCy POS-теггинг")

nlp_spacy = spacy.load('en_core_web_sm')

start_time = time.time()

spacy_results = []
for sent in test_sentences:
    doc = nlp_spacy(' '.join(sent))
    spacy_results.append([upos_tags.get(token.pos_, 17) for token in doc if token.text in sent])

spacy_time = time.time() - start_time

spacy_predicted_labels = [tag for sent in spacy_results for tag in sent]

spacy_true_labels = [tag for sent in test_labels for tag in sent]
min_length_spacy = min(len(spacy_true_labels), len(spacy_predicted_labels))
spacy_true_labels = spacy_true_labels[:min_length_spacy]
spacy_predicted_labels = spacy_predicted_labels[:min_length_spacy]

spacy_accuracy = accuracy_score(spacy_true_labels, spacy_predicted_labels)
spacy_precision, spacy_recall, spacy_f1, _ = precision_recall_fscore_support(spacy_true_labels, spacy_predicted_labels, average='macro')

# --- Stanza POS-теггинг ---
print("\nStanza POS-теггинг")

stanza.download('en')
nlp_stanza = stanza.Pipeline(lang='en', processors='tokenize,pos')

start_time = time.time()

stanza_results = []
for sent in test_sentences:
    doc = nlp_stanza(' '.join(sent))
    stanza_results.append([upos_tags.get(word.upos, 17) for sentence in doc.sentences for word in sentence.words])

stanza_time = time.time() - start_time

stanza_predicted_labels = [tag for sent in stanza_results for tag in sent]

stanza_true_labels = [tag for sent in test_labels for tag in sent]
min_length_stanza = min(len(stanza_true_labels), len(stanza_predicted_labels))
stanza_true_labels = stanza_true_labels[:min_length_stanza]
stanza_predicted_labels = stanza_predicted_labels[:min_length_stanza]

stanza_accuracy = accuracy_score(stanza_true_labels, stanza_predicted_labels)
stanza_precision, stanza_recall, stanza_f1, _ = precision_recall_fscore_support(stanza_true_labels, stanza_predicted_labels, average='macro')

print("\nРезультаты оценки моделей на основе Word2Vec и FastText:")
print(f"Word2Vec Skip-gram: Accuracy={acc_w2v_skipgram:.4f}, Precision={prec_w2v_skipgram:.4f}, Recall={rec_w2v_skipgram:.4f}, F1={f1_w2v_skipgram:.4f}, Time={time_w2v_skipgram:.2f} seconds")
print(f"Word2Vec CBOW: Accuracy={acc_w2v_cbow:.4f}, Precision={prec_w2v_cbow:.4f}, Recall={rec_w2v_cbow:.4f}, F1={f1_w2v_cbow:.4f}, Time={time_w2v_cbow:.2f} seconds")
print(f"FastText: Accuracy={acc_fasttext:.4f}, Precision={prec_fasttext:.4f}, Recall={rec_fasttext:.4f}, F1={f1_fasttext:.4f}, Time={time_fasttext:.2f} seconds")

print("\nРезультаты POS-теггеров:")

print("\nNLTK POS-теггер:")
print(f"Accuracy: {nltk_accuracy:.4f}, Precision: {nltk_precision:.4f}, Recall: {nltk_recall:.4f}, F1: {nltk_f1:.4f}, Time: {nltk_time:.2f} seconds")

print("\nSpaCy POS-теггер:")
print(f"Accuracy: {spacy_accuracy:.4f}, Precision: {spacy_precision:.4f}, Recall: {spacy_recall:.4f}, F1: {spacy_f1:.4f}, Time: {spacy_time:.2f} seconds")

print("\nStanza POS-теггер:")
print(f"Accuracy: {stanza_accuracy:.4f}, Precision: {stanza_precision:.4f}, Recall: {stanza_recall:.4f}, F1: {stanza_f1:.4f}, Time: {stanza_time:.2f} seconds")