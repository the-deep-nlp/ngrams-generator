# NGrams Generator

This ngrams package generates the unigram, bigram, trigram through different setttings. This supports the excerpts from several languages such as English, French, Spanish, Arabic and Portuguese.

# Install

```pip install git+https://github.com/the-deep-nlp/ngrams-generator.git```


# Usages

There are several parameters that can be set during the object instantiation.

from ngrams_generator import NGramsGenerator

ng = NGramsGenerator(
    max_ngrams_items: int,
    generate_unigrams: bool,
    generate_bigrams: bool,
    generate_trigrams: bool,
    enable_stopwords: bool,
    enable_stemming: bool,
    enable_case_sensitive: bool
)

ngram_tokens = ng(list_of_entries)


## Description of the parameters

`max_ngrams_items:` Maximum number of ngram tokens to publish

`generate_unigrams:` Whether to generate unigrams

`generate_bigrams:` Whether to generate bigrams

`generate_trigrams:` Whether to generate trigrams

`enable_stopwords:` Whether stopwords should be enabled

`enable_stemming:` Whether stemming should be enabled

`enable_case_sensitive:` Whether case sensitivity should be enabled
