import math
import preprocess_data
from itertools import chain

class NgramModel:
  def __init__(self, vocab: list[str], train: list[list[str]], test: list[list[str]]) -> None:
    self.vocab = vocab
    self.train = train
    self.test = test
  
  """
  - Fungsionalitas model ini adalah menghasilkan koleksi n-gram model.
  - Seperti 2-gram, artinya di dalam koleksi terdapat pasangan, seperti: '<s> saya', 'saya sedang', 'sedang makan', 'makan nasi', 'nasi kapau'.
  - Expected output berupa dictionary dengan pasangan key berupa n-length token serta value berupa kemunculan n-length token tersebut di dalam corpus.
  - Output format berupa dictionary.
  """
  def generate_n_grams(self, data: list[list[str]], n: int, start_token: str = '<s>', end_token='</s>') -> dict:
    # TODO: Implement based on the given description
    ngram_counts = {}  # Initialize a dictionary to store N-gram counts

    for sentence_tokens in data:
        sentence_tokens = [start_token] * (n - 1) + sentence_tokens + [end_token]  # Add start and end tokens
        ngrams = [tuple(sentence_tokens[i:i + n]) for i in range(len(sentence_tokens) - n + 1)]

        for ngram in ngrams:
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

    return ngram_counts
  
  """
  - Fungsionalitas method ini menghitung probabilitas suatu kata given kata/kumpulan kata.
  - Sederhananya, method ini merupakan implementasi dari ekspresi P(w_i|w_1:{i-1}).
  - Perlu diperhatikan bahwa pada parameter terdapat 'laplace_number' yang artinya Anda diharapkan mengimplementasikan add-one (laplace) smoothing.
  - Output format berupa float.
  """
  def count_probability(self, predicted_word: str, given_word: list[str], n_gram_counts, n_plus1_gram_counts, vocabulary_size, laplace_number: float = 1.0) -> float:
    # TODO: Implement based on the given description
    ngram = tuple(given_word + [predicted_word])
    ngram_count = n_gram_counts.get(ngram, 0)
    context = tuple(given_word)
    context_count = n_gram_counts.get(context, 0)

    prob = (ngram_count + laplace_number) / (context_count + (vocabulary_size * laplace_number))

    return prob
  
  """
  - Silakan Anda menggunakan method ini untuk bermain-main/menguji segala kemungkinan sentence/word generation berdasarkan method count_probability yang telah Anda bangun.
  """
  def probabilities_for_all_vocab(self, given_word: list[str], n_gram_counts, n_plus1_gram_counts, vocabulary, end_token='</s>', unknown_token='<unk>',  laplace_number=1.0):
    given_word = tuple(given_word)
    vocabulary = vocabulary + [end_token, unknown_token]
    vocab_size = len(vocabulary)

    probs = dict()
    for word in vocabulary:
      prob = self.count_probability(word, given_word, n_gram_counts, n_plus1_gram_counts, vocab_size, laplace_number=laplace_number)
      probs[word] = prob
    return probs

  """
  - Fungsionalitas pada method ini adalah untuk mengevaluasi n-gram model Anda menggunakan metrik perplexity.
  """
  def count_perplexity(self, sentence, n_gram_counts, n_plus1_gram_counts, vocab_size, start_token='<s>', end_token = '</s>', laplace_number=1.0):
    # TODO: Implement based on the given description
    sentence_tokens = [start_token] + sentence + [end_token]
    n = len(sentence_tokens)

    log_prob_sum = 0.0
    for i in range(n - 1):
        context = sentence_tokens[max(0, i - len(ngram) + 1):i]
        predicted_word = sentence_tokens[i]

        prob = self.count_probability(predicted_word, context, n_gram_counts, n_plus1_gram_counts, vocab_size, laplace_number=laplace_number)
        log_prob_sum += math.log2(prob)

    perplexity = 2 ** (-log_prob_sum / (n - 1))

    return perplexity

def main():
  """
  EXAMPLE:
  - Pada contoh ini menggunakan scenario no lowercasing (cased)
  """
  lowercase: bool = False
  preprocess = preprocess_data.Preprocess()
  vocab, train, test = preprocess.load_from_pickle(lowercase)

  model = NgramModel(vocab, train, test)
  flatten_test = list(chain.from_iterable(test))

  """
  EXAMPLE:
  - Silakan berkreasi se-kreatif mungkin menggunakan beragam n-gram model yang Anda inginkan.
  - Anda dibebaskan untuk mengganti/menambah/menghapus contoh kombinasi di bawah ini sesuai dengan kreativitas Anda.
  - Kami sangat menghargai kreativitas Anda terkait Tugas Individu 2 ini.
  """
  unigram_counts = model.generate_n_grams(train, 1)
  bigram_counts = model.generate_n_grams(train, 2)
  trigram_counts = model.generate_n_grams(train, 3)

  """
  EXAMPLE:
  - Di bawah ini merupakan contoh/cara untuk generate kalimat dari n-gram LM
  """

  """
  UNIGRAM
  """
  generate_S_unigram = model.probabilities_for_all_vocab(['saya'], train, unigram_counts, vocab)
  print(max(generate_S_unigram, key=generate_S_unigram.get))
  print(sorted(generate_S_unigram.items(), key=lambda x:x[1], reverse=True)[:5])

  """
  BIGRAM
  """
  generate_S_bigram = model.probabilities_for_all_vocab(['Berkas'], unigram_counts, bigram_counts, vocab)
  print(max(generate_S_bigram, key=generate_S_bigram.get))
  print(sorted(generate_S_bigram.items(), key=lambda x:x[1], reverse=True)[:5])

  """
  TRIGRAM
  """
  generate_S_trigram = model.probabilities_for_all_vocab(['pergi'], bigram_counts, trigram_counts, vocab)
  print(max(generate_S_trigram, key=generate_S_trigram.get))
  print(sorted(generate_S_trigram.items(), key=lambda x:x[1], reverse=True)[:5])


  """
  - Di bawah ini merupakan contoh/cara untuk menilai perplexity dari kalimat yang telah Anda generate dari langkah sebelumnya.
  """

  """
  UNIGRAM
  """
  perplexity_test_unigram = model.count_perplexity(flatten_test, train, unigram_counts, len(vocab), laplace_number=1.0)
  print(f"n = 1, Perplexity: {perplexity_test_unigram:.4f}")

  """
  BIGRAM
  """
  perplexity_test_bigram = model.count_perplexity(flatten_test, unigram_counts, bigram_counts, len(vocab), laplace_number=1.0)
  print(f"n = 1, Perplexity: {perplexity_test_bigram:.4f}")


  """
  TRIGRAM
  """
  perplexity_test_trigram = model.count_perplexity(flatten_test, bigram_counts, trigram_counts, len(vocab), laplace_number=1.0)
  print(f"n = 2, Perplexity: {perplexity_test_trigram:.4f}")

  """
  TRIGRAM from generated sentence
  """
  perplexity_test_random = model.count_perplexity(['<s>', 'cagar', 'budaya', 'merupakan', 'aset', 'di', 'indonesia', '</s>'], bigram_counts, trigram_counts, len(vocab), laplace_number=1.0)
  print(f"n = 2, Perplexity: {perplexity_test_random:.4f}")

if __name__ == "__main__":
  main()
