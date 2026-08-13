source("helpers.R")

check_unary_cases <- function(function_, cases) for (case in cases) stopifnot(
  function_(case[[1L]]) == case[[2L]]
)

soundex_cases <- list(
  c("Robert", "R163"), c("Rupert", "R163"), c("Rubin", "R150"),
  c("Ashcraft", "A261"), c("Ashcroft", "A261"), c("Tymczak", "T522"),
  c("Pfister", "P236"), c("Honeyman", "H555"), c("R", "R000"),
  c("Lee", "L000"), c("robert", "R163"), c("RUPERT", "R163"),
  c("Smith", "S530"), c("O'Brien", "O165"), c("Mc-Donald", "M235")
)
check_unary_cases(soundex, soundex_cases)
stopifnot(
  soundex("") == "", soundex("12345") == "", soundex(",,,") == "",
  soundex(" \n\t") == "", soundex("café") == soundex("cf"),
  soundex("Müller") == soundex("Mller"), soundex("你好世界") == "",
  soundex("Ashcraft") == "A261", soundex("Tymczak") == "T522",
  soundex("Bobcat") == "B123", soundex("Blackwell") == "B424",
  soundex("Strzelinski") == "S362"
)
bytes <- "Robert"
Encoding(bytes) <- "bytes"
stopifnot(soundex(bytes) == "R163")
expect_error(soundex(12345), "character")
stopifnot(
  soundex_equal("Robert", "Rupert"), soundex_equal("Smith", "Smyth"),
  soundex_equal("ashcraft", "Ashcroft"), !soundex_equal("Robert", "Smith"),
  !soundex_equal("Robert", ""), !soundex_equal("", ""),
  soundex_equal(bytes, "Rupert")
)

metaphone_cases <- list(
  c("Thompson", "0MPSN"), c("Catherine", "K0RN"), c("Kathryn", "K0RN"),
  c("Schmidt", "SKMTT"), c("Robert", "RBRT"), c("Smith", "SM0"),
  c("Caesar", "KSR"), c("Ashcraft", "AXKRFT"), c("Tymczak", "TMKSK"),
  c("Pfister", "PFSTR"), c("Honeyman", "HNMN"), c("Lloyd", "LT"),
  c("McDonald", "MKTNLT"), c("Hwang", "HWNK"), c("Williamson", "WLMSN"),
  c("Vandyke", "FNTK"), c("Phillips", "FLPS"), c("Xavier", "SFR"),
  c("Zhang", "SHNK"), c("Yamaguchi", "YMKX"), c("Bobcat", "BBKT"),
  c("Blackwell", "BLKWL"), c("Aether", "E0R"), c("Gnaeus", "NS"),
  c("Knight", "NT"), c("Wright", "RT"), c("Lamb", "LM"),
  c("Match", "MX"), c("Chess", "XS"), c("Quiche", "KX"),
  c("Hugh", "H"), c("Through", "0R"), c("Tough", "T"), c("Cough", "K"),
  c("Plough", "PL"), c("Ghost", "KHST"), c("Caught", "KT"), c("Hour", "HR")
)
check_unary_cases(metaphone, metaphone_cases)
stopifnot(
  metaphone("") == "", metaphone("12345") == "", metaphone("   ") == "",
  metaphone("ROBERT") == metaphone("robert"),
  metaphone("café") == metaphone("cf"), metaphone("你好") == "",
  metaphone_equal("Catherine", "Kathryn"),
  !metaphone_equal("Robert", "Smith"), !metaphone_equal("", ""),
  metaphone_equal("ROBERT", "robert")
)
expect_error(metaphone(12345), "character")
for (name in c("Schmidt", "Hugh", "Through", "Wright", "Caught", "Ghost")) stopifnot(
  metaphone(name) == metaphone(name, variant = MetaphoneVariant$PHILIPS)
)
variant_cases <- list(
  c("Schmidt", "SKMTT", "SXMTT"), c("Hugh", "H", "HKH"),
  c("Through", "0R", "0RKH"), c("Tough", "T", "TKH"),
  c("Cough", "K", "KKH"), c("Plough", "PL", "PLKH"),
  c("Robert", "RBRT", "RBRT"), c("Catherine", "K0RN", "K0RN"),
  c("Knight", "NT", "NT"), c("Wright", "RT", "RT"),
  c("Caught", "KT", "KT"), c("Ghost", "KHST", "KHST")
)
for (case in variant_cases) stopifnot(
  metaphone(case[[1L]], MetaphoneVariant$PHILIPS) == case[[2L]],
  metaphone(case[[1L]], MetaphoneVariant$JELLYFISH) == case[[3L]]
)
stopifnot(
  metaphone_equal("Hugh", "Hue", MetaphoneVariant$PHILIPS),
  !metaphone_equal("Hugh", "Hue", MetaphoneVariant$JELLYFISH)
)

nysiis_cases <- list(
  c("Watkins", "WATCAN"), c("Wilkins", "WALCAN"), c("Wilkinson", "WALCANSAN"),
  c("Robert", "RABAD"), c("Smith", "SNAT"), c("Schmidt", "SNAD"),
  c("Catherine", "CATARAN"), c("Kathryn", "CATRYN"), c("Knight", "NAGT"),
  c("Wright", "WRAGT"), c("Caesar", "CASAR"), c("Macarthur", "MCARTAR"),
  c("Phillips", "FALAP"), c("Pfister", "FASTAR"), c("Schwartz", "SWART"),
  c("Hawthorne", "HATARN"), c("Yarborough", "YARBARAG"), c("Thompson", "TANPSAN"),
  c("Anderson", "ANDARSAN"), c("Johnson", "JANSAN"), c("Williams", "WALAN"),
  c("Hugh", "HAG"), c("Through", "TRAG"), c("Caught", "CAGT"),
  c("Plough", "PLAG"), c("Lloyd", "LAYD"), c("Honeyman", "HANAYNAN"),
  c("McDonald", "MCDANALD"), c("Sarah", "SAR"), c("Michael", "MACAL"),
  c("Hannah", "HAN"), c("Joshua", "JAS"), c("William", "WALAN")
)
check_unary_cases(nysiis, nysiis_cases)
stopifnot(
  nysiis("") == "", nysiis("12345") == "", nysiis("ROBERT") == nysiis("robert"),
  nysiis("café") == nysiis("caf"), nysiis("你好") == "",
  nysiis_equal("Catherine", "Catharine"), nysiis_equal("Watkins", "Watkin"),
  !nysiis_equal("Robert", "Smith"), !nysiis_equal("Robert", "Roberto"),
  !nysiis_equal("Smith", "Smyth"), !nysiis_equal("", "")
)
expect_error(nysiis(12345), "character")

match_rating_cases <- list(
  c("Robert", "RBRT"), c("Rupert", "RPRT"), c("Smith", "SMTH"),
  c("Smyth", "SMYTH"), c("Schmidt", "SCHMDT"), c("Catherine", "CTHRN"),
  c("Kathryn", "KTHRYN"), c("Christopher", "CHRPHR"), c("Williams", "WLMS"),
  c("Williamson", "WLMSN"), c("Jonathan", "JNTHN"), c("John", "JHN"),
  c("Hannah", "HNH"), c("Sarah", "SRH"), c("Michael", "MCHL"),
  c("Watkins", "WTKNS"), c("Wilkinson", "WLKNSN"), c("Lloyd", "LYD"),
  c("Honeyman", "HNYMN"), c("McDonald", "MCDNLD"), c("Caesar", "CSR"),
  c("Aether", "ATHR"), c("Knight", "KNGHT"), c("Wright", "WRGHT")
)
check_unary_cases(match_rating_codex, match_rating_cases)
stopifnot(
  match_rating_codex("") == "", match_rating_codex("12345") == "",
  match_rating_codex("ROBERT") == match_rating_codex("robert"),
  match_rating_codex("café") == match_rating_codex("caf"),
  match_rating_codex("你好") == "", match_rating_codex("Oscar") == "OSCR",
  nchar(match_rating_codex("Williamsonburg")) == 6L
)
expect_error(match_rating_codex(12345), "character")
for (pair in list(
  c("Robert", "Rupert"), c("Smith", "Smyth"), c("Catherine", "Kathryn"),
  c("Williams", "Williamson"), c("Knight", "Night"), c("John", "Jon"),
  c("Sarah", "Sara"), c("Christopher", "Chris"), c("Lloyd", "Floyd"),
  c("Hannah", "Hana")
)) stopifnot(match_rating_compare(pair[[1L]], pair[[2L]]))
for (pair in list(c("Robert", "Smith"), c("Catherine", "Robert"), c("Sarah", "Robert"))) stopifnot(
  !match_rating_compare(pair[[1L]], pair[[2L]])
)
stopifnot(
  !match_rating_compare("", "Robert"), !match_rating_compare("Robert", ""),
  !match_rating_compare("", ""),
  match_rating_compare("Robert", "Rupert") == match_rating_compare("Rupert", "Robert"),
  match_rating_compare("Smith", "Schmidt") == match_rating_compare("Schmidt", "Smith")
)

caverphone_cases <- list(
  c("Stevenson", "STFNSN1111"), c("Peter", "PTA1111111"), c("Peady", "PTA1111111"),
  c("Thompson", "TMPSN11111"), c("Smith", "SMT1111111"), c("Schmidt", "SKMT111111"),
  c("add", "AT11111111"), c("aid", "AT11111111"), c("at", "AT11111111"),
  c("art", "AT11111111"), c("eat", "AT11111111"), c("earth", "AT11111111"),
  c("head", "AT11111111"), c("hit", "AT11111111"), c("hot", "AT11111111"),
  c("hold", "AT11111111"), c("hard", "AT11111111"), c("heart", "AT11111111"),
  c("it", "AT11111111"), c("out", "AT11111111"), c("old", "AT11111111"),
  c("mb", "M111111111"), c("mbmb", "MPM1111111"), c("rather", "RTA1111111"),
  c("ready", "RTA1111111"), c("writer", "RTA1111111"), c("social", "SSA1111111"),
  c("able", "APA1111111"), c("appear", "APA1111111"), c("Cailean", "KLN1111111")
)
check_unary_cases(caverphone, caverphone_cases)
stopifnot(
  caverphone("") == "1111111111", caverphone("12345") == "1111111111",
  caverphone("STEVENSON") == caverphone("stevenson"),
  caverphone("café") == caverphone("caf"), caverphone("你好") == "1111111111"
)
expect_error(caverphone(12345), "character")
stopifnot(all(nchar(caverphone(c("a", "Stevenson", "Christopher", "Antidisestablishmentarianism"))) == 10L))

double_cases <- list(
  c("Smith", "SM0", "XMT"), c("Robert", "RPRT", ""), c("Knight", "NT", ""),
  c("Wright", "RT", ""), c("Caesar", "SSR", ""), c("Lloyd", "LT", ""),
  c("Lamb", "LMP", ""), c("Match", "MX", ""), c("Chess", "XS", ""),
  c("Honeyman", "HNMN", ""), c("Schmidt", "XMT", "SMT"),
  c("Schwartz", "XRTS", "XFRTS"), c("Pawlowski", "PLSK", "PLFSK"),
  c("Hugh", "H", ""), c("High", "H", ""), c("Tough", "TF", ""),
  c("Cough", "KF", ""), c("Gnaeus", "NS", ""), c("Tymczak", "TMSK", "TMXK")
)
for (case in double_cases) {
  value <- double_metaphone(case[[1L]])
  stopifnot(value$primary[[1L]] == case[[2L]], value$alternate[[1L]] == case[[3L]])
}
stopifnot(
  double_metaphone("")$primary[[1L]] == "",
  double_metaphone("12345")$primary[[1L]] == "",
  identical(double_metaphone("SMITH"), double_metaphone("smith")),
  identical(double_metaphone("Müller"), double_metaphone("Mller"))
)
expect_error(double_metaphone(12345), "character")
short <- double_metaphone("Christopher", max_length = 4)
full <- double_metaphone("Christopher")
stopifnot(
  nchar(short$primary) <= 4, nchar(short$alternate) <= 4,
  nchar(full$primary) > 4, startsWith(full$primary, short$primary),
  double_metaphone("Robert")$alternate[[1L]] == ""
)
python_hugh <- double_metaphone("Hugh", variant = DoubleMetaphoneVariant$PYTHON)
python_high <- double_metaphone("High", variant = DoubleMetaphoneVariant$PYTHON)
stopifnot(
  python_hugh$primary[[1L]] == "HH", python_hugh$alternate[[1L]] == "",
  python_high$primary[[1L]] == "HH",
  identical(
    double_metaphone("Smith", variant = DoubleMetaphoneVariant$PYTHON),
    double_metaphone("Smith")
  ),
  identical(
    double_metaphone("Hugh"),
    double_metaphone("Hugh", variant = DoubleMetaphoneVariant$COMMONS)
  ),
  DoubleMetaphoneVariant$COMMONS == 0L,
  DoubleMetaphoneVariant$PYTHON == 1L
)

cologne_cases <- list(
  c("Wikipedia", "3412"), c("Breschnew", "17863"), c("Müller", "657"),
  c("Mueller", "657"), c("Schmidt", "862"), c("Schneider", "8627"),
  c("Fischer", "387"), c("Weber", "317"), c("Meyer", "67"), c("Mayer", "67"),
  c("Maier", "67"), c("Meier", "67"), c("Wagner", "3467"), c("Becker", "147"),
  c("Schulz", "858"), c("Hoffmann", "0366"), c("Heinz", "068"),
  c("Müllers", "6578"), c("Köln", "456"), c("Großmann", "47866")
)
check_unary_cases(cologne_phonetic, cologne_cases)
stopifnot(
  cologne_phonetic("") == "", cologne_phonetic("12345") == "",
  cologne_phonetic("MÜLLER") == cologne_phonetic("müller"),
  cologne_phonetic("Müller") == cologne_phonetic("Mueller"),
  cologne_phonetic("Köln") == cologne_phonetic("Koeln"),
  cologne_phonetic("Süß") == cologne_phonetic("Suess"),
  cologne_phonetic("Aaron") == "076", cologne_phonetic("Wikipedia") == "3412",
  cologne_phonetic("Müller") == "657"
)
expect_error(cologne_phonetic(12345), "character")

daitch_cases <- list(
  c("Straßburg", "294795"), c("Strasburg", "294795"), c("Éregon", "095600"),
  c("Eregon", "095600"), c("AKSSOL", "054800"),
  c("GERSCHFELD", "547830|545783|594783|594578"),
  c("AUERBACH", "097400|097500"), c("OHRBACH", "097400|097500"),
  c("LIPSHITZ", "874400"), c("LIPPSZYC", "874400|874500"),
  c("LEWINSKY", "876450"), c("LEVINSKI", "876450"),
  c("SZLAMAWICZ", "486740"), c("SHLAMOVITZ", "486740"),
  c("Washington", "746536"), c("GOLDEN", "583600"),
  c("Alpert", "087930"), c("Breuer", "791900")
)
check_unary_cases(daitch_mokotoff, daitch_cases)
first <- daitch_mokotoff("GERSCHFELD", branching = FALSE)
stopifnot(
  !grepl("|", first, fixed = TRUE), nchar(first) == 6L,
  daitch_mokotoff(" \t\n\r Washington \t\n\r ") == daitch_mokotoff("Washington"),
  daitch_mokotoff("Éregon") == daitch_mokotoff("Eregon"),
  all(nchar(strsplit(daitch_mokotoff("Lee"), "|", fixed = TRUE)[[1L]]) == 6L),
  daitch_mokotoff("") == "000000"
)

# ASCII CE_BYTES checks shared by every Python bytes-input test.
for (function_ in list(metaphone, nysiis, match_rating_codex, caverphone, cologne_phonetic, daitch_mokotoff)) {
  value <- "Robert"
  Encoding(value) <- "bytes"
  stopifnot(identical(function_(value), function_("Robert")))
}

# Python cases: all 81 native tests in tests/test_phonetic.py. Each
# parameterized canonical table is copied above; bytes cases use R's CE_BYTES
# representation, and return-shape checks use character vectors/data frames.
