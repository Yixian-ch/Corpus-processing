import json
from argparse import ArgumentParser
from matplotlib import pyplot as plt
import numpy as np

def get_args():
    """Parse command line arguments"""
    parser = ArgumentParser()
    parser.add_argument("files", nargs='+', help="JSON files to analyze")
    args = parser.parse_args()
    return args

def parse_json(args):
    """Extract descriptions from JSON files"""
    corpus = ""
    for filename in args.files:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            if 'description' in item:
                corpus += item["description"] + " "
    return corpus

def statistic_test(corpus):
    """Analyze word frequency distribution and test Zipf's law"""
    # Preprocess text
    words_raw = corpus.lower().split()
    
    # Load stopwords
    with open("stopwords-fr.txt") as f:
        stopwords = set(f.read().split())
    
    # Filter words: keep only alphabetic words not in stopwords
    words = [word for word in words_raw if word.isalpha() and word not in stopwords]
    
    print(f"Total words: {len(words)}")
    
    # Count word frequencies
    word_count = {}
    for word in words:
        word_count[word] = word_count.get(word, 0) + 1
    
    # Sort by frequency
    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    
    print(f"Unique words: {len(word_count)}")
    print("\nZipf's Law Test (top 20 words):")
    print("Rank | Word           | Frequency | Expected (Zipf)")
    print("-" * 55)
    
    # Analyze top words
    max_freq = sorted_words[0][1]
    ranks = []
    frequencies = []
    expected_freqs = []
    
    for rank, (word, freq) in enumerate(sorted_words[:20], 1):
        expected_freq = max_freq / rank
        print(f"{rank:4d} | {word:14s} | {freq:9d} | {expected_freq:8.1f}")
        ranks.append(rank)
        frequencies.append(freq)
        expected_freqs.append(expected_freq)
    
    # Calculate Zipf conformity
    total_ratio = sum(freq / (max_freq / rank) 
                     for rank, (_, freq) in enumerate(sorted_words[:10], 1))
    avg_ratio = total_ratio / 10
    
    print(f"\nAverage ratio (actual/expected): {avg_ratio:.2f}")
    
    if 0.8 < avg_ratio < 1.2:
        print("The corpus approximately follows Zipf's law")
    else:
        print("The corpus deviates from Zipf's law")
    
    # Create visualization
    plot_zipf(ranks, frequencies, expected_freqs, sorted_words)
    
    return word_count

def plot_zipf(ranks, frequencies, expected_freqs, sorted_words):
    """Visualize Zipf's law distribution"""
    plt.figure(figsize=(12, 5))
    
    # Log-log plot
    plt.subplot(1, 2, 1)
    plt.loglog(ranks, frequencies, 'bo-', label='Observed', markersize=8)
    plt.loglog(ranks, expected_freqs, 'r--', label='Zipf theoretical', linewidth=2)
    plt.xlabel('Rank (log scale)')
    plt.ylabel('Frequency (log scale)')
    plt.title("Zipf's Law Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Bar chart for top 10 words
    plt.subplot(1, 2, 2)
    top_10 = sorted_words[:10]
    words = [w for w, _ in top_10]
    freqs = frequencies[:10]
    expected = expected_freqs[:10]
    
    x = np.arange(len(words))
    width = 0.35
    
    plt.bar(x - width/2, freqs, width, label='Observed', color='blue', alpha=0.7)
    plt.bar(x + width/2, expected, width, label='Expected', color='red', alpha=0.7)
    
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Top 10 Most Frequent Words')
    plt.xticks(x, words, rotation=45, ha='right')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('zipf_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def nlg_corpus_tests(corpus, word_count):
    """Evaluate corpus suitability for NLG model training"""
    print("\n" + "="*60)
    print("NLG CORPUS EVALUATION")
    print("="*60)
    
    words = corpus.lower().split()
    
    # 1. Corpus size analysis
    print("\n1. CORPUS SIZE")
    print(f"Total tokens: {len(words):,}")
    print(f"Unique words: {len(word_count):,}")
    print(f"Type-token ratio: {len(word_count)/len(words):.4f}")
    
    
    # 2. Sentence analysis
    sentences = [s.strip() for s in corpus.split('.') if s.strip()]
    lengths = [len(s.split()) for s in sentences]
    
    print("\n2. SENTENCE ANALYSIS")
    print(f"Sentences: {len(lengths)}")
    print(f"Average length: {np.mean(lengths):.1f} words")
    print(f"Std deviation: {np.std(lengths):.1f}")
    print(f"Min/Max: {min(lengths)}/{max(lengths)}")
    
    # 3. Vocabulary coverage
    print("\n3. VOCABULARY COVERAGE")
    sorted_freq = sorted(word_count.values(), reverse=True)
    total = sum(sorted_freq)
    
    for n in [100, 500, 1000, 5000]:
        if n < len(sorted_freq):
            coverage = sum(sorted_freq[:n]) / total * 100
            print(f"Top {n} words: {coverage:.1f}% of corpus")
    
    # 4. Rare words analysis
    hapax = sum(1 for freq in word_count.values() if freq == 1)
    rare = sum(1 for freq in word_count.values() if freq < 5)
    
    print("\n4. RARE WORDS")
    print(f"Hapax legomena: {hapax} ({hapax/len(word_count)*100:.1f}%)")
    print(f"Rare words (<5): {rare} ({rare/len(word_count)*100:.1f}%)")
    
    # 5. Lexical diversity
    print("\n5. LEXICAL DIVERSITY")
    segment_size = 1000
    ttrs = []
    
    for i in range(0, len(words) - segment_size, segment_size):
        segment = words[i:i + segment_size]
        ttr = len(set(segment)) / len(segment)
        ttrs.append(ttr)
    
    if ttrs:
        print(f"Average TTR: {np.mean(ttrs):.3f}")
        print(f"TTR std deviation: {np.std(ttrs):.3f}")
    
def main():
    args = get_args()
    corpus = parse_json(args)
    word_count = statistic_test(corpus)
    nlg_corpus_tests(corpus, word_count)

if __name__ == "__main__":
    main()