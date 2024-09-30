import requests
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import spacy

# Load the small English language model
nlp = spacy.load("en_core_web_sm")

class SequencedNLPAnalyzer:
    def __init__(self):
        self.data = {}

    def preprocess_text(self, text):
        """
        Preprocess text by tokenizing and removing stopwords using SpaCy.
        :param text: Input text string.
        :return: List of processed tokens.
        """
        doc = nlp(text)
        # Return tokens that are alphabetic and not stopwords
        tokens = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop]
        return tokens

    # The rest of the class can remain the same...

class SequencedNLPAnalyzer:
    def __init__(self):
        self.data = {}  # Dictionary to hold text data by year and policy type
        self.api_base_url = "https://api.github.com/repos/PlatformGovernanceArchive/pga-corpus/contents/"
        self.stop_words = set(stopwords.words('english'))  # Load NLTK stop words

    def fetch_data(self, platform, start_year, end_year):
        """
        Fetch data from the GitHub repository for a given platform and date range.

        :param platform: Platform name, e.g., 'Facebook'.
        :param start_year: Start year for data filtering, e.g., '2010'.
        :param end_year: End year for data filtering, e.g., '2020'.
        """
        self.data = {}  # Reset the data dictionary
        policies = ['Community Guidelines', 'Privacy Policy', 'Terms of Service']
        for policy in policies:
            # Construct the path for each policy document type
            path = f"Versions/Markdown/{platform}/{policy.replace(' ', '%20')}"
            url = f"{self.api_base_url}{path}"
            response = requests.get(url)
            if response.status_code == 200:
                files = response.json()  # List of files in the directory
                for file in files:
                    if file['name'].endswith('.md'):  # Only consider Markdown files
                        year = file['name'][:4]  # Extract the year from the filename
                        if start_year <= year <= end_year:  # Check if the year is within range
                            content = self.get_file_content(file['download_url'])
                            if year not in self.data:
                                self.data[year] = {}  # Create year entry if not exists
                            self.data[year][policy] = self.preprocess_text(content)
            yield f"Fetched data for {platform} - {policy}"
        yield "Data fetching complete"

    def get_file_content(self, url):
        """
        Get the content of a file from a given URL.

        :param url: URL to the file.
        :return: Content of the file as a string.
        """
        response = requests.get(url)
        return response.text if response.status_code == 200 else ""

    def preprocess_text(self, text):
        """
        Preprocess text by tokenizing, lowercasing, and removing stopwords.

        :param text: Input text string.
        :return: List of processed tokens.
        """
        # Tokenize and remove non-alphabetic characters
        tokens = [word.lower() for word in word_tokenize(text) if word.isalpha()]
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        return tokens

    def keyword_in_context(self, keyword, window_size=5):
        """
        Extract keyword in context from the processed text.

        :param keyword: Keyword to search for in the text.
        :param window_size: Number of words to include on each side of the keyword.
        :return: Dictionary with years as keys and keyword context as values.
        """
        kwic_results = defaultdict(list)
        for year, policies in self.data.items():
            for policy, tokens in policies.items():
                keyword_indices = [i for i, token in enumerate(tokens) if token == keyword]
                for index in keyword_indices:
                    left_context = ' '.join(tokens[max(0, index-window_size):index])
                    right_context = ' '.join(tokens[index+1:min(len(tokens), index+window_size+1)])
                    kwic_results[year].append((left_context, keyword, right_context))
        return kwic_results

    def word_frequency_analysis(self):
        """
        Perform word frequency analysis on the processed text.

        :return: Dictionary with years as keys and word frequency DataFrames as values.
        """
        word_freq_results = {}
        for year, policies in self.data.items():
            all_tokens = []
            for policy, tokens in policies.items():
                all_tokens.extend(tokens)  # Aggregate tokens across policies
            freq_dist = FreqDist(all_tokens)
            word_freq_results[year] = pd.DataFrame(freq_dist.most_common(20), columns=['Word', 'Frequency'])
        return word_freq_results

    def plot_word_frequencies(self, word_freq_results):
        """
        Plot word frequencies for each year.

        :param word_freq_results: Dictionary of word frequency DataFrames.
        """
        for year, freq_df in word_freq_results.items():
            plt.figure(figsize=(10, 6))
            plt.barh(freq_df['Word'], freq_df['Frequency'], color='skyblue')
            plt.title(f'Word Frequency Analysis - {year}')
            plt.xlabel('Frequency')
            plt.ylabel('Word')
            plt.gca().invert_yaxis()  # Invert y-axis to display the highest frequency at the top
            plt.show()

    def display_kwic_results(self, kwic_results):
        """
        Display the keyword-in-context results.

        :param kwic_results: Dictionary with years as keys and context tuples as values.
        """
        for year, contexts in kwic_results.items():
            print(f"\nKeyword in Context for Year: {year}")
            for left, keyword, right in contexts:
                print(f"...{left} {keyword} {right}...")

# Example Usage
if __name__ == "__main__":
    # Initialize the analyzer
    analyzer = SequencedNLPAnalyzer()

    # Fetch data for Facebook from 2010 to 2020
    for status in analyzer.fetch_data('Facebook', '2010', '2020'):
        print(status)

    # Perform Keyword-in-Context Analysis
    kwic_results = analyzer.keyword_in_context('policy')
    analyzer.display_kwic_results(kwic_results)

    # Perform Word Frequency Analysis
    word_freq_results = analyzer.word_frequency_analysis()

    # Plot Word Frequencies
    analyzer.plot_word_frequencies(word_freq_results)
