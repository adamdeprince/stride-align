\name{phonetic_api}
\alias{MetaphoneVariant}
\alias{DoubleMetaphoneVariant}
\alias{BmpmRuleType}
\alias{soundex}
\alias{soundex_equal}
\alias{metaphone}
\alias{metaphone_equal}
\alias{nysiis}
\alias{nysiis_equal}
\alias{match_rating_codex}
\alias{match_rating_compare}
\alias{caverphone}
\alias{cologne_phonetic}
\alias{daitch_mokotoff}
\alias{double_metaphone}
\alias{beider_morse}
\title{Phonetic encoders and comparisons}
\usage{
soundex(input)
soundex_equal(query, target)
metaphone(input, variant = MetaphoneVariant$PHILIPS)
metaphone_equal(query, target, variant = MetaphoneVariant$PHILIPS)
nysiis(input)
nysiis_equal(query, target)
match_rating_codex(input)
match_rating_compare(query, target)
caverphone(input)
cologne_phonetic(input)
daitch_mokotoff(input, branching = TRUE, folding = TRUE)
double_metaphone(input, max_length = 64,
  variant = DoubleMetaphoneVariant$COMMONS)
beider_morse(input, rule_type = BmpmRuleType$APPROX,
  concat = TRUE, max_phonemes = 20)
}
\arguments{
\item{input}{A character vector to encode.}
\item{query, target}{Character vectors to compare phonetically.}
\item{variant}{A value from the corresponding variant catalog.}
\item{branching}{Whether Daitch-Mokotoff emits alternative encodings.}
\item{folding}{Whether Daitch-Mokotoff folds accented letters.}
\item{max_length}{Maximum Double Metaphone code length.}
\item{rule_type}{Approximate or exact Beider-Morse rules.}
\item{concat}{Whether Beider-Morse concatenates multiword input.}
\item{max_phonemes}{Maximum Beider-Morse alternatives.}
}
\value{
Encoders return character vectors; equality functions return logical vectors.
\code{double_metaphone()} returns a data frame with \code{primary} and
\code{alternate} columns.
}
\description{
Vectorized native phonetic algorithms corresponding to the Python top-level
stride-align API.
}
