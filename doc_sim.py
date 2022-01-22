# HW-3
# Web Crawler
import math
import time

import bs4
import nltk
import numpy as np
import requests
from bs4 import BeautifulSoup
from nltk import PorterStemmer
from scipy.spatial import distance


def read_page(url, text_list, title_list):
    seed_url_page = requests.get(url)
    seed_text = seed_url_page.text
    toc_first_occurence = seed_text.find("id=\"toc\"")
    main_content_text = seed_text[:toc_first_occurence]
    soup = BeautifulSoup(main_content_text, 'html.parser')
    main_content_tags = soup.find_all("p")
    final_str = ""
    for tag in main_content_tags:
        final_str += tag.get_text()
    text_list.append(final_str)
    # Get titles for pages
    title_list.append(soup.find(id="firstHeading").string)
    return text_list


def get_first_ten_links(url):
    link_list = []
    url_page = requests.get(url)
    page_text = url_page.text
    soup2 = BeautifulSoup(page_text, 'html.parser')
    links = soup2.find_all("p")
    count = 0
    for link in links:
        if count == 10:
            break
        for obj in link:
            if type(obj) is bs4.element.Tag:
                if obj.has_attr("href") and "wiki" in obj["href"] and ".svg" not in obj["href"]:
                    if obj["href"] not in link_list:
                        if count != 10:
                            link_list.append(obj["href"])
                            count = count + 1
                        else:
                            break
    return link_list


def check_reciprocal_link(link, check_link, check_reciprocal_list):
    url_page = requests.get(link)
    page_text = url_page.text
    soup3 = BeautifulSoup(page_text, 'html.parser')
    links = soup3.find_all("a", href=True)
    for sub_link in links:
        if check_link == ("https://en.wikipedia.org" + str(sub_link["href"])):
            check_reciprocal_list.append(True)
            return check_reciprocal_list
    check_reciprocal_list.append(False)
    return check_reciprocal_list


def stopword_removal(tokens):
    stopwords = ["I", "i", "in", "a", "is", "who", "about", "it", "will", "an", "of", "with", "are", "on", "the", "as",
                 "or", "www", "at", "that", "be", "by", "this", "com", "to", "for", "was", "from", "what", "how",
                 "when", "where", "\'\'", "\'", "\\"]
    for word in stopwords:
        tokens = [token for token in tokens if token != word]
    return tokens


def text_stopword_removal(text):
    stopwords = ["I", "i", "in", "a", "is", "who", "about", "it", "will", "an", "of", "with", "are", "on", "the", "as",
                 "or", "www", "at", "that", "be", "by", "this", "com", "to", "for", "was", "from", "what", "how",
                 "when", "where"]
    text_tokens = nltk.word_tokenize(text)
    for word in stopwords:
        text_tokens = [token for token in text_tokens if token != word]
    cleaned_text = ' '.join(token for token in text_tokens)
    return cleaned_text


def token_stemming(cleaned_tokens):
    ps = PorterStemmer()
    for i in range(len(cleaned_tokens)):
        cleaned_tokens[i] = ps.stem(cleaned_tokens[i])
    return cleaned_tokens


def text_stemming(text):
    ps = PorterStemmer()
    stemmed_text = ' '.join(ps.stem(token) for token in nltk.word_tokenize(text))
    return stemmed_text


if __name__ == "__main__":
    seed_page = input("Enter seed page URL: ")
    webpage_text_list = []
    title_list = []
    webpage_text_list = read_page(seed_page, webpage_text_list, title_list)
    webpage_link_list = get_first_ten_links(seed_page)
    seed_link = seed_page
    check_list = []
    for web_link in webpage_link_list:
        web_link = "https://en.wikipedia.org" + web_link
        check_list = check_reciprocal_link(web_link, seed_link, check_list)
        time.sleep(5)  # For politeness, give gap of 5 secs
        webpage_text_list = read_page(web_link, webpage_text_list, title_list)
    # Tokenizing
    token_list = []
    for page_text in webpage_text_list:
        token_list.append(nltk.word_tokenize(page_text.lower()))
    # Stopword removal
    cleaned_list = []
    for page_tokens in token_list:
        cleaned_list.append(stopword_removal(page_tokens))
    # Stemming
    stemmed_list = []
    for cleaned_tokens in cleaned_list:
        stemmed_list.append(token_stemming(cleaned_tokens))
    # Creating vocabulary
    vocab = set()
    for stemmed_token_list in stemmed_list:
        for stemmed_token in stemmed_token_list:
            vocab.add(stemmed_token)
    # Stemming and stopword removal for intro text for all pages
    webpage_stemmed_text_list = []
    for webpage_text in webpage_text_list:
        cleaned_text = text_stopword_removal(webpage_text.lower())
        webpage_stemmed_text_list.append(text_stemming(cleaned_text))
    # TF calculations
    init_matrix = np.zeros((len(webpage_stemmed_text_list), len(vocab)))
    tf_matrix = np.zeros((len(webpage_stemmed_text_list), len(vocab)))
    set_elem = list(vocab)
    for row_num in range(0, len(webpage_stemmed_text_list)):
        for col_num in range(0, len(set_elem)):
            tf = webpage_stemmed_text_list[row_num].count(" " + set_elem[col_num] + " ")
            init_matrix[row_num][col_num] = tf
            if tf == 0:
                tf_matrix[row_num][col_num] = 0
            else:
                tf_matrix[row_num][col_num] = (math.log10(tf) + 1)
    # IDF calculations
    idf_matrix = np.count_nonzero(init_matrix, axis=0)
    idf_matrix_final = np.zeros(idf_matrix.shape, dtype=float)
    for idf_col in range(0, idf_matrix.shape[0]):
        idf_matrix_final[idf_col] = math.log10((len(webpage_link_list) + 1) / idf_matrix[idf_col])
    for col in range(0, idf_matrix_final.shape[0]):
        tf_matrix[:, col] = tf_matrix[:, col] * idf_matrix_final[col]
    # Cosine Similarity
    cosine_scores = []
    for row_num in range(1, tf_matrix.shape[0]):
        if np.count_nonzero(tf_matrix[row_num]) == 0:
            cosine_scores.append(0)
        else:
            cosine_scores.append(1 - distance.cosine(tf_matrix[0], tf_matrix[row_num]))
    # Print results
    print()
    print("Seed Page: " + str(seed_page))
    print()
    print("Child Page, Cosine Similarity, Reciprocal Links")
    print()
    for print_num in range(0, len(cosine_scores)):
        print(str(title_list[print_num + 1]) + ", " + str(cosine_scores[print_num]) + ", " + str(check_list[print_num]))
