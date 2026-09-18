#pragma once

#include <string>
#include <vector>
#include <unordered_map>

namespace sweep {

struct Token {
    std::string text;
    std::string lemma;
    std::string pos;  // part of speech
    std::string ner;  // named entity tag
    int start = 0;
    int end = 0;
};

struct Entity {
    std::string text;
    std::string label;  // PERSON, ORG, LOC, DATE, etc.
    int start = 0;
    int end = 0;
    double confidence = 0;

    int length() const { return static_cast<int>(text.size()); }
};

struct NLPResult {
    std::vector<Token> tokens;
    std::vector<Entity> entities;
    std::vector<std::string> sentences;
    std::vector<std::string> keywords;
    std::string language;
};

std::vector<std::string> tokenize_words(const std::string& text);
std::vector<std::string> tokenize_sentences(const std::string& text);
std::vector<Token> tokenize_full(const std::string& text);
std::vector<Entity> extract_entities(const std::string& text);
std::vector<Entity> extract_emails(const std::string& text);
std::vector<Entity> extract_phones(const std::string& text);
std::vector<Entity> extract_urls(const std::string& text);
std::vector<Entity> extract_dates(const std::string& text);
std::vector<std::string> extract_keywords(const std::string& text, int top_n = 10);
double text_similarity(const std::string& a, const std::string& b);
std::string summarize(const std::string& text, int max_sentences = 3);
std::string sentiment(const std::string& text);
std::string stem(const std::string& word);
std::string lemmatize(const std::string& word);
std::string to_lower(const std::string& text);
std::string remove_stopwords(const std::string& text);
std::string normalize_text(const std::string& text);

struct TFIDFResult {
    std::vector<std::string> vocabulary;
    std::vector<std::vector<double>> matrix;
};

TFIDFResult compute_tfidf(const std::vector<std::string>& documents);
std::vector<double> vectorize_text(const std::string& text, const std::vector<std::string>& vocabulary);

}  // namespace sweep
