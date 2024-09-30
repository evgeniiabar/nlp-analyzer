from flask import Flask, render_template, request
import pandas as pd
import requests
import spacy
from collections import defaultdict

# Initialize Flask app
app = Flask(__name__)

# Load SpaCy English model
nlp = spacy.load('en_core_web_sm')


class SequencedNLPAnalyzer:
    def __init__(self):
        self.data = {}
        self.api_base_url = "https://api.github.com/repos/PlatformGovernanceArchive/pga-corpus/contents/"

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
        Preprocess text using SpaCy by tokenizing and removing stopwords and punctuation.

        :param text: Input text string.
        :return: List of processed tokens.
        """
        # Create a SpaCy document object
        doc = nlp(text)
        # Filter out stopwords and punctuation, and convert to lowercase
        tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
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
                    left_context = ' '.join(tokens[max(0, index - window_size):index])
                    right_context = ' '.join(tokens[index + 1:min(len(tokens), index + window_size + 1)])
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
            freq_dist = pd.Series(all_tokens).value_counts()
            word_freq_results[year] = pd.DataFrame(freq_dist.head(20), columns=['Frequency']).reset_index()
            word_freq_results[year].columns = ['Word', 'Frequency']
        return word_freq_results


# Initialize the analyzer
analyzer = SequencedNLPAnalyzer()


@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    keyword = ''
    if request.method == 'POST':
        # Get input parameters from the form
        platform = request.form.get('platform')
        start_year = request.form.get('start_year')
        end_year = request.form.get('end_year')
        keyword = request.form.get('keyword')

        # Fetch data for the specified platform and date range
        for status in analyzer.fetch_data(platform, start_year, end_year):
            print(status)  # Optional: Print status to the console

        # Perform Keyword-in-Context Analysis
        kwic_results = analyzer.keyword_in_context(keyword)

        # Perform Word Frequency Analysis
        word_freq_results = analyzer.word_frequency_analysis()

        # Prepare results for rendering
        results = {'kwic': kwic_results, 'frequency': word_freq_results}

    return render_template('index.html', results=results, keyword=keyword)


if __name__ == '__main__':
    app.run(debug=True)
