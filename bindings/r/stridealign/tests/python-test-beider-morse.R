source("helpers.R")

stopifnot(
  is.function(beider_morse), BmpmRuleType$APPROX == 0L,
  BmpmRuleType$EXACT == 1L, beider_morse("") == "",
  beider_morse("   ") == ""
)
bytes <- "Smith"
Encoding(bytes) <- "bytes"
stopifnot(beider_morse(bytes) == beider_morse("Smith"))
stopifnot(identical(beider_morse("Cohen"), beider_morse("Cohen")))
approx <- beider_morse("Renault", rule_type = BmpmRuleType$APPROX)
exact <- beider_morse("Renault", rule_type = BmpmRuleType$EXACT)
stopifnot(approx != exact, lengths(strsplit(approx, "|", fixed = TRUE)) >= lengths(strsplit(exact, "|", fixed = TRUE)))

cases <- list(
  list("Renault", "rinD|rinDlt|rina|rinalt|rino|rinolt|rinu|rinult", BmpmRuleType$APPROX, TRUE, 10),
  list("SntJohn-Smith", "sntjonsmit", BmpmRuleType$EXACT, TRUE, 10),
  list("d'ortley", "(ortlaj|ortlej)-(dortlaj|dortlej)", BmpmRuleType$EXACT, TRUE, 10),
  list(
    "van helsing",
    paste0(
      "(elSink|elsink|helSink|helsink|helzink|xelsink)-",
      "(banhelsink|fanhelsink|fanhelzink|vanhelsink|vanhelzink|vanjelsink)"
    ),
    BmpmRuleType$EXACT, FALSE, 10
  ),
  list(
    "Judenburg",
    "iudnbYrk|iudnbirk|iudnburk|xudnbirk|xudnburk|zudnbirk|zudnburk",
    BmpmRuleType$APPROX, TRUE, 10
  )
)
for (case in cases) stopifnot(beider_morse(
  case[[1L]], rule_type = case[[3L]], concat = case[[4L]],
  max_phonemes = case[[5L]]
) == case[[2L]])
full <- beider_morse("Schwarzenegger", max_phonemes = 20)
capped <- beider_morse("Schwarzenegger", max_phonemes = 2)
stopifnot(
  lengths(strsplit(full, "|", fixed = TRUE)) >= lengths(strsplit(capped, "|", fixed = TRUE)),
  lengths(strsplit(capped, "|", fixed = TRUE)) <= 2L
)

# Python cases: test_beider_morse_imported,
# test_empty_input_returns_empty_string, test_bytes_input_supported,
# test_identical_inputs_produce_identical_codes,
# test_distinct_rule_types_differ, test_apache_commons_codec_vectors,
# test_max_phonemes_cap_respected.
