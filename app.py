from flask import Flask, render_template, request, send_file
import pandas as pd
import requests
import spacy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import io

# Initialize Flask app
app = Flask(__name__)

# Load SpaCy English model
nlp = spacy.load('en_core_web_sm')


class SequencedNLPAnalyzer:
    def __init__(self):
        self.data = {}
        self.api_base_url = "https://api.github.com/repos/PlatformGovernanceArchive/pga-corpus/contents/"

    def fetch_data(self, platform, start_year, end_year, selected_policies):
        """
        Fetch data from the GitHub repository for a given platform and date range.
        Filters the data based on the selected policy types.
        """
        self.data = {}  # Reset the data dictionary
        self.fetched_policies = []  # Reset the fetched policies list

        for policy in selected_policies:  # Iterate over the selected policies only
            # Construct the path for each selected policy document type
            path = f"Versions/Markdown/{platform}/{policy.replace(' ', '%20')}"
            url = f"{self.api_base_url}{path}"
            response = requests.get(url)
            if response.status_code == 200:
                self.fetched_policies.append(policy)  # Track which policies were fetched
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
        """
        response = requests.get(url)
        return response.text if response.status_code == 200 else ""

    def preprocess_text(self, text):
        """
        Preprocess text using SpaCy by tokenizing and removing stopwords and punctuation.
        """
        # Create a SpaCy document object
        doc = nlp(text)
        # Filter out stopwords and punctuation, and convert to lowercase
        tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
        return tokens

    def keyword_in_context(self, keyword, window_size=5):
        """
        Extract keyword in context from the processed text.
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

    def word_frequency_analysis(self, keyword):
        """
        Perform word frequency analysis on the processed text, specifically for a given keyword.
        Returns a dictionary with year as the key and frequency as the value.
        """
        word_freq_by_year = {}
        for year, policies in self.data.items():
            # Aggregate all tokens for the year
            all_tokens = []
            for policy, tokens in policies.items():
                all_tokens.extend(tokens)

            # Count the occurrences of the specified keyword
            word_count = all_tokens.count(keyword.lower())
            word_freq_by_year[year] = word_count  # Store the count for the year

            print(f"Year {year} - {keyword} frequency: {word_count}")  # Debugging print statement

        return word_freq_by_year

    def generate_yearly_barplot(self, yearly_data, keyword):
        """
        Generate a bar plot showing the frequency of a keyword per year.
        """
        # Check if data is empty or None
        if not yearly_data:
            print("No yearly data to plot.")
            return None

        # Convert the yearly data dictionary to a DataFrame for plotting
        df = pd.DataFrame(list(yearly_data.items()), columns=['Year', 'Frequency'])

        # Sort the DataFrame by year
        df = df.sort_values(by='Year')

        print(f"Generating bar plot for yearly frequency of '{keyword}':\n{df}")  # Debug statement

        # Create the bar plot
        plt.figure(figsize=(10, 6))
        plt.bar(df['Year'], df['Frequency'], color='skyblue')
        plt.xlabel('Year')
        plt.ylabel('Frequency')
        plt.title(f'Frequency of "{keyword}" Per Year')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Save the plot to a BytesIO object
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()
        return img


# Initialize the analyzer
analyzer = SequencedNLPAnalyzer()


@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    keyword = ''
    img_path = None  # Initialize image path variable
    fetched_policies = []  # Variable to store fetched policies

    if request.method == 'POST':
        platform = request.form.get('platform')
        start_year = request.form.get('start_year')
        end_year = request.form.get('end_year')
        keyword = request.form.get('keyword')

        # Get the list of selected policies from the form
        selected_policies = request.form.getlist('policies')  # List of selected policies

        # Print the selected policies for debugging purposes
        print(f"Selected policies: {selected_policies}")

        # Fetch the data based on selected policies
        for status in analyzer.fetch_data(platform, start_year, end_year, selected_policies):
            print(status)  # Optional: Print status to the console for debugging

        fetched_policies = analyzer.fetched_policies  # Get the fetched policies from the analyzer

        # Analyze keyword in context and frequency over the years
        kwic_results = analyzer.keyword_in_context(keyword)
        yearly_word_freq = analyzer.word_frequency_analysis(keyword)  # Pass the keyword argument

        # Generate bar plot for word frequency per year
        barplot_img = analyzer.generate_yearly_barplot(yearly_word_freq, keyword)

        # If `barplot_img` is not None, save the image
        if barplot_img:
            img_path = os.path.join('static', 'yearly_frequency.png')
            with open(img_path, 'wb') as f:
                f.write(barplot_img.read())  # Write image content
            print(f"Bar plot saved successfully at {img_path}.")  # Debug message
        else:
            print("Yearly frequency bar plot image generation failed. No image will be saved.")

        results = {'kwic': kwic_results, 'yearly_frequency': yearly_word_freq, 'fetched_policies': fetched_policies}

    return render_template('index.html', results=results, keyword=keyword, img_path=img_path,
                           fetched_policies=fetched_policies)


@app.route('/static/<filename>')
def serve_image(filename):
    return send_file(f'static/{filename}', mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)
