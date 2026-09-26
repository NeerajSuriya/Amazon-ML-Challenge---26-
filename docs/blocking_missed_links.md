# Missed-link investigation for default blocking

## Scope and conclusion

This investigation compares the 52 ground-truth links absent from the default union
with the candidate output from `CandidateBlocker(BlockingConfig())` on the sample
data. The default union remains unchanged. The baseline is 161,162 candidate pairs
and retrieves 10,392 of 10,444 true links (99.5021% blocking recall).

All 52 misses are country-compatible. None has an exact normalized name or exact
normalized address, and none is an exact-key bucket overflow. The common cause is the
candidate-side frequency cutoff of 50: the links share only generic/high-frequency
name or address tokens, or share no token in that field. This is a data/blocking-key
coverage issue, not a country-filter issue.

## Miss categories

| Category | Count | Definition |
|---|---:|---|
| Address shared tokens over frequency | 40 | At least one shared address token, but every shared address token occurs more than 50 times among S2/S3 candidates; no informative address token reaches the default index. |
| Name shared tokens over frequency | 9 | At least one shared name token, but every shared name token occurs more than 50 times; the address is empty or has no shared token. |
| Both shared token sets over frequency | 3 | Both name and address have shared tokens, but all shared tokens in each field exceed its frequency cutoff. |
| **Total** | **52** | |

### Rule-level exclusion counts

- `country_name_exact`: 52/52 fail because `name_norm` is not exactly equal; 0 are rejected by country mismatch and 0 are rejected by the 500-record exact-bucket cap.
- `country_address_exact`: 52/52 fail because `address_norm` is not exactly equal; 0 are rejected by country mismatch and 0 are rejected by the 500-record exact-bucket cap.
- `name_token`: 43/52 have no shared normalized name token; the remaining 9 have shared name tokens but none with candidate frequency ≤50.
- `address_token`: 12/52 have no shared normalized address token; the remaining 40 have shared address tokens but none with candidate frequency ≤50.
- `name_address_token`: 52/52 fail because the pair does not have an informative name-token intersection and an informative address-token intersection under the defaults.
- Country filtering excludes 0/52. Missing addresses explain why several name-only misses cannot be recovered by address rules.

## Recovery-rule experiments

Each row below is evaluated as `baseline union ∪ proposed rule`; `added` counts only
new candidate identities beyond the 161,162-pair baseline. Frequency caps are inclusive.
Character n-gram rules use the processed `*_char_ngrams` lists. Prefix rules use a
prefix index with a 50-record bucket cap. Approximate rules use RapidFuzz top-5
retrieval per S1 record and are included only as a controlled comparison, not as an
all-pairs fuzzy join.

| Proposed rule | Misses recovered | Added | Total candidates | Recall |
|---|---:|---:|---:|---:|
| `Default union (control)` | 0 | 0 | 161,162 | 99.5021% |
| `Name token cap 75` | 7 | 11,423 | 172,585 | 99.5691% |
| `Name token cap 100` | 9 | 19,306 | 180,468 | 99.5883% |
| `Name token cap 200` | 10 | 91,010 | 252,172 | 99.5979% |
| `Name token cap 500` | 11 | 170,378 | 331,540 | 99.6074% |
| `Address token cap 75` | 13 | 68,242 | 229,404 | 99.6266% |
| `Address token cap 100` | 20 | 128,603 | 289,765 | 99.6936% |
| `Address token cap 200` | 35 | 482,340 | 643,502 | 99.8372% |
| `Address token cap 500` | 42 | 1,284,863 | 1,446,025 | 99.9043% |
| `Address token cap 1000` | 43 | 2,046,194 | 2,207,356 | 99.9138% |
| `Address overlap ≥2 tokens, no frequency cap` | 41 | 448,003 | 609,165 | 99.8947% |
| `Address overlap ≥2 tokens, cap 200` | 23 | 47,008 | 208,170 | 99.7223% |
| `Address char n-gram cap 50` | 19 | 279,261 | 440,423 | 99.6840% |
| `Address char n-gram cap 100` | 34 | 902,699 | 1,063,861 | 99.8277% |
| `Address char n-gram cap 200` | 42 | 2,434,805 | 2,595,967 | 99.9043% |
| `Address prefix length 3, bucket cap 50` | 15 | 17,551 | 178,713 | 99.6457% |
| `Address prefix length 4, bucket cap 50` | 12 | 4,845 | 166,007 | 99.6170% |
| `Name char n-gram cap 200` | 16 | 1,233,658 | 1,394,820 | 99.6553% |
| `Country-independent address token cap 50` | 0 | 18,892 | 180,054 | 99.5021% |
| `Country-independent name token cap 50` | 0 | 15,489 | 176,651 | 99.5021% |
| `Approx address RapidFuzz ≥70, top 5` | 2 | 187 | 161,349 | 99.5213% |
| `Approx address RapidFuzz ≥80, top 5` | 1 | 4 | 161,166 | 99.5117% |
| `Approx name RapidFuzz ≥70, top 5` | 6 | 3,573 | 164,735 | 99.5596% |
| `Approx name RapidFuzz ≥80, top 5` | 3 | 1,754 | 162,916 | 99.5308% |
| `Approx name RapidFuzz ≥90, top 5` | 2 | 166 | 161,328 | 99.5213% |

### Combined alternatives

| Combination (baseline plus) | Misses recovered | Added | Total candidates | Recall |
|---|---:|---:|---:|---:|
| Address overlap ≥2, unlimited token frequency + name token cap 100 | 49 | 466,499 | 627,661 | 99.9713% |
| Address overlap ≥2 + address token cap 200 + name token cap 150 | 52 | 800,909 | 962,071 | 100.0000% |
| Address token cap 1000 + name token cap 150 | 52 | 2,098,059 | 2,259,221 | 100.0000% |

### Interpretation

- Relaxing address frequency is the most productive single change, but it is expensive: cap 1000 recovers 43 misses while adding 2,046,194 candidates.
- Requiring at least two shared address tokens without a frequency cap recovers 41 misses for 448,003 added candidates. It uses joint evidence, but still admits generic city/state/number combinations.
- A cap-200 two-address-token rule is much smaller (47,008 added candidates) but recovers only 23 misses.
- Address character n-grams recover 42 misses at cap 200, but add 2,434,805 candidates; they are not a cheap drop-in replacement on this sample.
- Name-token relaxation has modest recovery because only nine misses are name-token-only; cap 100 recovers nine while adding 19,306 candidates.
- Country-independent rules recover none of the 52 because none is a country mismatch. They add 15,489 (name) and 18,892 (address) candidates and therefore have no recall justification here.
- Prefix rules recover at most 15 misses in the tested configurations. Controlled top-5 approximate name/address retrieval recovers at most six misses and does not solve the address-frequency problem.
- The smallest tested full-recall combination is still approximately 6x the baseline candidate volume. It should not be enabled by default without downstream precision/latency measurements.

## Per-link evidence

For every row, token frequencies are counts across all non-empty S2/S3 candidate
records. `—` means an empty list or missing value. “Default exclusion” gives the
exact reason each current rule did not emit the pair. All displayed country pairs are
compatible under the current missing-country policy.

### 1. `S1-118502154` → `S3-592354415` (S3)

- **Country:** `india` ↔ `india`; compatible, no country exclusion.
- **Name:** `north constructions private limited ↔ न र थ क स ट रक श स private limited`
- **Address:** `office number bw9021 maharashtra 9th floor g block bharat diamond bourse bkc bandra east bandra suburban mumbai ↔ office number bwa 9021 bombay मह र ष ट र`
- **S1 name tokens:** constructions, limited, north, private
- **Candidate name tokens:** limited, private, क, ट, थ, न, र, रक, श, स
- **Shared name tokens (frequency):** limited=1751, private=1512
- **S1 address tokens:** 9th, bandra, bharat, bkc, block, bourse, bw9021, diamond, east, floor, g, maharashtra, mumbai, number, office, suburban
- **Candidate address tokens:** 9021, bombay, bwa, number, office, ट, मह, र, ष
- **Shared address tokens (frequency):** number=1199, office=124
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → shared name tokens all exceed frequency 50; `address_token` → shared address tokens all exceed frequency 50; `name_address_token` → no informative token intersection in both fields.
- **Category:** `both_shared_token_sets_over_frequency`.

### 2. `S1-119033930` → `S3-45249084` (S3)

- **Country:** `india` ↔ `india`; compatible, no country exclusion.
- **Name:** `royal technology private limited ↔ र यल ट क न ल ज प र इव ट ल म ट ड`
- **Address:** `g 8 a number 1 bhagirath palace north east delhi 1737 ↔ 737 north east delhi द ल ल`
- **S1 name tokens:** limited, private, royal, technology
- **Candidate name tokens:** इव, क, ज, ट, ड, न, प, म, यल, र, ल
- **Shared name tokens (frequency):** —
- **S1 address tokens:** 1, 1737, 8, a, bhagirath, delhi, east, g, north, number, palace
- **Candidate address tokens:** 737, delhi, east, north, द, ल
- **Shared address tokens (frequency):** delhi=497, east=205, north=456
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → no shared name token; `address_token` → shared address tokens all exceed frequency 50; `name_address_token` → no informative token intersection in both fields.
- **Category:** `address_shared_tokens_over_frequency`.

### 3. `S1-119033930` → `S3-745743078` (S3)

- **Country:** `india` ↔ `india`; compatible, no country exclusion.
- **Name:** `royal technology private limited ↔ र यल ट क न ल ज प र इव ट ल म ट ड`
- **Address:** `g 8 a number 1 bhagirath palace north east delhi 1737 ↔ 737 north east delhi द ल ल`
- **S1 name tokens:** limited, private, royal, technology
- **Candidate name tokens:** इव, क, ज, ट, ड, न, प, म, यल, र, ल
- **Shared name tokens (frequency):** —
- **S1 address tokens:** 1, 1737, 8, a, bhagirath, delhi, east, g, north, number, palace
- **Candidate address tokens:** 737, delhi, east, north, द, ल
- **Shared address tokens (frequency):** delhi=497, east=205, north=456
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → no shared name token; `address_token` → shared address tokens all exceed frequency 50; `name_address_token` → no informative token intersection in both fields.
- **Category:** `address_shared_tokens_over_frequency`.

### 4. `S1-139707378` → `S2-504459617` (S2)

- **Country:** `india` ↔ `india`; compatible, no country exclusion.
- **Name:** `ace consultancy llp ↔ ಏಸ್ ಕನ್ಸಲ್ಟೆನ್ಸಿ ಎಲ್ಎಲ್ ಪಿ`
- **Address:** `number 1 khata no 147 thubrahalli village ramagondanahalli post whitefield main r oad bangalore karnataka ↔ number 1 bangalore bengaluru karnataka`
- **S1 name tokens:** ace, consultancy, llp
- **Candidate name tokens:** ಎಲ್ಎಲ್, ಏಸ್, ಕನ್ಸಲ್ಟೆನ್ಸಿ, ಪಿ
- **Shared name tokens (frequency):** —
- **S1 address tokens:** 1, 147, bangalore, karnataka, khata, main, no, number, oad, post, r, ramagondanahalli, thubrahalli, village, whitefield
- **Candidate address tokens:** 1, bangalore, bengaluru, karnataka, number
- **Shared address tokens (frequency):** 1=599, bangalore=226, karnataka=110, number=1199
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → no shared name token; `address_token` → shared address tokens all exceed frequency 50; `name_address_token` → no informative token intersection in both fields.
- **Category:** `address_shared_tokens_over_frequency`.

### 5. `S1-139707378` → `S2-733469786` (S2)

- **Country:** `india` ↔ `india`; compatible, no country exclusion.
- **Name:** `ace consultancy llp ↔ ಏಸ್ ಕನ್ಸಲ್ಟೆನ್ಸಿ ಎಲ್ಎಲ್ ಪಿ`
- **Address:** `number 1 khata no 147 thubrahalli village ramagondanahalli post whitefield main r oad bangalore karnataka ↔ door number 1 ಕರ್ನಾಟಕ bengaluru`
- **S1 name tokens:** ace, consultancy, llp
- **Candidate name tokens:** ಎಲ್ಎಲ್, ಏಸ್, ಕನ್ಸಲ್ಟೆನ್ಸಿ, ಪಿ
- **Shared name tokens (frequency):** —
- **S1 address tokens:** 1, 147, bangalore, karnataka, khata, main, no, number, oad, post, r, ramagondanahalli, thubrahalli, village, whitefield
- **Candidate address tokens:** 1, bengaluru, door, number, ಕರ್ನಾಟಕ
- **Shared address tokens (frequency):** 1=599, number=1199
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → no shared name token; `address_token` → shared address tokens all exceed frequency 50; `name_address_token` → no informative token intersection in both fields.
- **Category:** `address_shared_tokens_over_frequency`.

### 6. `S1-144406637` → `S3-711709706` (S3)

- **Country:** `india` ↔ `india`; compatible, no country exclusion.
- **Name:** `technologies swarn infra private limited ↔ technologiesswarninfra com`
- **Address:** `plot number 1113 road number 54 jubilee hills hyderabad telangana ↔ 1304 hyderabad urban hyderabad త ల గ ణ`
- **S1 name tokens:** infra, limited, private, swarn, technologies
- **Candidate name tokens:** com, technologiesswarninfra
- **Shared name tokens (frequency):** —
- **S1 address tokens:** 1113, 54, hills, hyderabad, jubilee, number, plot, road, telangana
- **Candidate address tokens:** 1304, hyderabad, urban, గ, ణ, త, ల
- **Shared address tokens (frequency):** hyderabad=176
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → no shared name token; `address_token` → shared address tokens all exceed frequency 50; `name_address_token` → no informative token intersection in both fields.
- **Category:** `address_shared_tokens_over_frequency`.

### 7. `S1-158009900` → `S3-125070393` (S3)

- **Country:** `us` ↔ `us`; compatible, no country exclusion.
- **Name:** `pediatric partners associates ↔ pediatric partners associates incorporated`
- **Address:** `10 glen ellen boulevard millis ma ↔ <empty>`
- **S1 name tokens:** associates, partners, pediatric
- **Candidate name tokens:** associates, incorporated, partners, pediatric
- **Shared name tokens (frequency):** associates=134, partners=198, pediatric=77
- **S1 address tokens:** 10, boulevard, ellen, glen, ma, millis
- **Candidate address tokens:** —
- **Shared address tokens (frequency):** —
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → shared name tokens all exceed frequency 50; `address_token` → no shared address token; `name_address_token` → no informative token intersection in both fields.
- **Category:** `name_shared_tokens_over_frequency`.

### 8. `S1-164081713` → `S2-940155859` (S2)

- **Country:** `us` ↔ `us`; compatible, no country exclusion.
- **Name:** `zh acquisition incorporated ↔ zhacquisition com`
- **Address:** `30445 1st place federal way wa ↔ first pl federla way wa`
- **S1 name tokens:** acquisition, incorporated, zh
- **Candidate name tokens:** com, zhacquisition
- **Shared name tokens (frequency):** —
- **S1 address tokens:** 1st, 30445, federal, place, wa, way
- **Candidate address tokens:** federla, first, pl, wa, way
- **Shared address tokens (frequency):** wa=96, way=148
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → no shared name token; `address_token` → shared address tokens all exceed frequency 50; `name_address_token` → no informative token intersection in both fields.
- **Category:** `address_shared_tokens_over_frequency`.

### 9. `S1-200453197` → `S2-842417062` (S2)

- **Country:** `india` ↔ `india`; compatible, no country exclusion.
- **Name:** `health processing limited ↔ health pbrocessign limited`
- **Address:** `opposite water park danapur dinapur cum khagaul bihar patna khagaul road ↔ <empty>`
- **S1 name tokens:** health, limited, processing
- **Candidate name tokens:** health, limited, pbrocessign
- **Shared name tokens (frequency):** health=101, limited=1751
- **S1 address tokens:** bihar, cum, danapur, dinapur, khagaul, opposite, park, patna, road, water
- **Candidate address tokens:** —
- **Shared address tokens (frequency):** —
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → shared name tokens all exceed frequency 50; `address_token` → no shared address token; `name_address_token` → no informative token intersection in both fields.
- **Category:** `name_shared_tokens_over_frequency`.

### 10. `S1-224096874` → `S2-710381516` (S2)

- **Country:** `us` ↔ `us`; compatible, no country exclusion.
- **Name:** `dental center pllc ↔ dental pllc center`
- **Address:** `2035 swanson meadows road bonner mt ↔ <empty>`
- **S1 name tokens:** center, dental, pllc
- **Candidate name tokens:** center, dental, pllc
- **Shared name tokens (frequency):** center=468, dental=55, pllc=62
- **S1 address tokens:** 2035, bonner, meadows, mt, road, swanson
- **Candidate address tokens:** —
- **Shared address tokens (frequency):** —
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → shared name tokens all exceed frequency 50; `address_token` → no shared address token; `name_address_token` → no informative token intersection in both fields.
- **Category:** `name_shared_tokens_over_frequency`.

### 11. `S1-24424354` → `S2-817827446` (S2)

- **Country:** `india` ↔ `india`; compatible, no country exclusion.
- **Name:** `bharat foundation private limited ↔ भारत फाउंडेशन प्राइवेट लिमिटेड`
- **Address:** `e 11 ram path shyam nagar extension new sanganer road jaipur rajasthan jaipur ↔ e 11 jaipur rajasthan`
- **S1 name tokens:** bharat, foundation, limited, private
- **Candidate name tokens:** प्राइवेट, फाउंडेशन, भारत, लिमिटेड
- **Shared name tokens (frequency):** —
- **S1 address tokens:** 11, e, extension, jaipur, nagar, new, path, rajasthan, ram, road, sanganer, shyam
- **Candidate address tokens:** 11, e, jaipur, rajasthan
- **Shared address tokens (frequency):** 11=72, e=97, jaipur=57, rajasthan=54
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → no shared name token; `address_token` → shared address tokens all exceed frequency 50; `name_address_token` → no informative token intersection in both fields.
- **Category:** `address_shared_tokens_over_frequency`.

### 12. `S1-24424354` → `S3-862089762` (S3)

- **Country:** `india` ↔ `india`; compatible, no country exclusion.
- **Name:** `bharat foundation private limited ↔ भ रत फ उ ड शन प र इव ट ल म ट ड`
- **Address:** `e 11 ram path shyam nagar extension new sanganer road jaipur rajasthan jaipur ↔ e 1 1 jaipur null rj`
- **S1 name tokens:** bharat, foundation, limited, private
- **Candidate name tokens:** इव, उ, ट, ड, प, फ, भ, म, र, रत, ल, शन
- **Shared name tokens (frequency):** —
- **S1 address tokens:** 11, e, extension, jaipur, nagar, new, path, rajasthan, ram, road, sanganer, shyam
- **Candidate address tokens:** 1, e, jaipur, null, rj
- **Shared address tokens (frequency):** e=97, jaipur=57
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → no shared name token; `address_token` → shared address tokens all exceed frequency 50; `name_address_token` → no informative token intersection in both fields.
- **Category:** `address_shared_tokens_over_frequency`.

### 13. `S1-25922331` → `S2-261106760` (S2)

- **Country:** `india` ↔ `india`; compatible, no country exclusion.
- **Name:** `apex constructions private limited ↔ एपेक्स कंस्ट्रक्शंस प्राइवेट लिमिटेड`
- **Address:** `b 7 31 1 block b 7 safdarjung enclave delhi south delhi delhi ↔ b 7 31 1 south delhi delhi`
- **S1 name tokens:** apex, constructions, limited, private
- **Candidate name tokens:** एपेक्स, कंस्ट्रक्शंस, प्राइवेट, लिमिटेड
- **Shared name tokens (frequency):** —
- **S1 address tokens:** 1, 31, 7, b, block, delhi, enclave, safdarjung, south
- **Candidate address tokens:** 1, 31, 7, b, delhi, south
- **Shared address tokens (frequency):** 1=599, 31=53, 7=186, b=384, delhi=497, south=227
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → no shared name token; `address_token` → shared address tokens all exceed frequency 50; `name_address_token` → no informative token intersection in both fields.
- **Category:** `address_shared_tokens_over_frequency`.

### 14. `S1-268581045` → `S2-211106281` (S2)

- **Country:** `india` ↔ `india`; compatible, no country exclusion.
- **Name:** `surya engineering private limited ↔ ಸೂರ್ಯ ಇಂಜಿನಿಯರಿಂಗ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್`
- **Address:** `number 7 bellary road gangenahalli bangalore karnataka ↔ karnataka bangalore number 7`
- **S1 name tokens:** engineering, limited, private, surya
- **Candidate name tokens:** ಇಂಜಿನಿಯರಿಂಗ್, ಪ್ರೈವೇಟ್, ಲಿಮಿಟೆಡ್, ಸೂರ್ಯ
- **Shared name tokens (frequency):** —
- **S1 address tokens:** 7, bangalore, bellary, gangenahalli, karnataka, number, road
- **Candidate address tokens:** 7, bangalore, karnataka, number
- **Shared address tokens (frequency):** 7=186, bangalore=226, karnataka=110, number=1199
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → no shared name token; `address_token` → shared address tokens all exceed frequency 50; `name_address_token` → no informative token intersection in both fields.
- **Category:** `address_shared_tokens_over_frequency`.

### 15. `S1-268581045` → `S3-109505025` (S3)

- **Country:** `india` ↔ `india`; compatible, no country exclusion.
- **Name:** `surya engineering private limited ↔ ಸ ರ ಯ ಇ ಜ ನ ಯರ ಗ ಪ ರ ವ ಟ ಲ ಮ ಟ ಡ`
- **Address:** `number 7 bellary road gangenahalli bangalore karnataka ↔ number 7 bangalore ka`
- **S1 name tokens:** engineering, limited, private, surya
- **Candidate name tokens:** ಇ, ಗ, ಜ, ಟ, ಡ, ನ, ಪ, ಮ, ಯ, ಯರ, ರ, ಲ, ವ, ಸ
- **Shared name tokens (frequency):** —
- **S1 address tokens:** 7, bangalore, bellary, gangenahalli, karnataka, number, road
- **Candidate address tokens:** 7, bangalore, ka, number
- **Shared address tokens (frequency):** 7=186, bangalore=226, number=1199
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → no shared name token; `address_token` → shared address tokens all exceed frequency 50; `name_address_token` → no informative token intersection in both fields.
- **Category:** `address_shared_tokens_over_frequency`.

### 16. `S1-268581045` → `S3-610446810` (S3)

- **Country:** `india` ↔ `india`; compatible, no country exclusion.
- **Name:** `surya engineering private limited ↔ ಸ ರ ಯ ಇ ಜ ನ ಯರ ಗ ಪ ರ ವ ಟ ಲ ಮ ಟ ಡ`
- **Address:** `number 7 bellary road gangenahalli bangalore karnataka ↔ number 7 bangalore ka`
- **S1 name tokens:** engineering, limited, private, surya
- **Candidate name tokens:** ಇ, ಗ, ಜ, ಟ, ಡ, ನ, ಪ, ಮ, ಯ, ಯರ, ರ, ಲ, ವ, ಸ
- **Shared name tokens (frequency):** —
- **S1 address tokens:** 7, bangalore, bellary, gangenahalli, karnataka, number, road
- **Candidate address tokens:** 7, bangalore, ka, number
- **Shared address tokens (frequency):** 7=186, bangalore=226, number=1199
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → no shared name token; `address_token` → shared address tokens all exceed frequency 50; `name_address_token` → no informative token intersection in both fields.
- **Category:** `address_shared_tokens_over_frequency`.

### 17. `S1-268581045` → `S3-701012115` (S3)

- **Country:** `india` ↔ `india`; compatible, no country exclusion.
- **Name:** `surya engineering private limited ↔ ಸ ರ ಯ ಇ ಜ ನ ಯರ ಗ ಪ ರ ವ ಟ ಲ ಮ ಟ ಡ`
- **Address:** `number 7 bellary road gangenahalli bangalore karnataka ↔ door number 7 bangalore ka`
- **S1 name tokens:** engineering, limited, private, surya
- **Candidate name tokens:** ಇ, ಗ, ಜ, ಟ, ಡ, ನ, ಪ, ಮ, ಯ, ಯರ, ರ, ಲ, ವ, ಸ
- **Shared name tokens (frequency):** —
- **S1 address tokens:** 7, bangalore, bellary, gangenahalli, karnataka, number, road
- **Candidate address tokens:** 7, bangalore, door, ka, number
- **Shared address tokens (frequency):** 7=186, bangalore=226, number=1199
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → no shared name token; `address_token` → shared address tokens all exceed frequency 50; `name_address_token` → no informative token intersection in both fields.
- **Category:** `address_shared_tokens_over_frequency`.

### 18. `S1-26969687` → `S3-449345466` (S3)

- **Country:** `india` ↔ `india`; compatible, no country exclusion.
- **Name:** `ilead storage private limited ↔ ileadstorage com`
- **Address:** `no 5 sy no 90 3 harikumar layout manjunathnagar nagasandra post tumkur road bangalore karnataka ↔ number 5 peenya ಕರ ನ ಟಕ`
- **S1 name tokens:** ilead, limited, private, storage
- **Candidate name tokens:** com, ileadstorage
- **Shared name tokens (frequency):** —
- **S1 address tokens:** 3, 5, 90, bangalore, harikumar, karnataka, layout, manjunathnagar, nagasandra, no, post, road, sy, tumkur
- **Candidate address tokens:** 5, number, peenya, ಕರ, ಟಕ, ನ
- **Shared address tokens (frequency):** 5=190
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → no shared name token; `address_token` → shared address tokens all exceed frequency 50; `name_address_token` → no informative token intersection in both fields.
- **Category:** `address_shared_tokens_over_frequency`.

### 19. `S1-288786162` → `S2-244746627` (S2)

- **Country:** `india` ↔ `india`; compatible, no country exclusion.
- **Name:** `lotus global limited ↔ লোটাস গ্লোবাল লিমিটেড`
- **Address:** `7 13 narkel danga main road kolkata howrah west bengal ↔ 007 13 kolkata west bengal`
- **S1 name tokens:** global, limited, lotus
- **Candidate name tokens:** গ্লোবাল, লিমিটেড, লোটাস
- **Shared name tokens (frequency):** —
- **S1 address tokens:** 13, 7, bengal, danga, howrah, kolkata, main, narkel, road, west
- **Candidate address tokens:** 007, 13, bengal, kolkata, west
- **Shared address tokens (frequency):** 13=79, bengal=131, kolkata=191, west=455
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → no shared name token; `address_token` → shared address tokens all exceed frequency 50; `name_address_token` → no informative token intersection in both fields.
- **Category:** `address_shared_tokens_over_frequency`.

### 20. `S1-288786162` → `S3-974285783` (S3)

- **Country:** `india` ↔ `india`; compatible, no country exclusion.
- **Name:** `lotus global limited ↔ ল ট স গ ল ব ল ল ম ট ড`
- **Address:** `7 13 narkel danga main road kolkata howrah west bengal ↔ 7 13 kolkata null wb`
- **S1 name tokens:** global, limited, lotus
- **Candidate name tokens:** গ, ট, ড, ব, ম, ল, স
- **Shared name tokens (frequency):** —
- **S1 address tokens:** 13, 7, bengal, danga, howrah, kolkata, main, narkel, road, west
- **Candidate address tokens:** 13, 7, kolkata, null, wb
- **Shared address tokens (frequency):** 13=79, 7=186, kolkata=191
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → no shared name token; `address_token` → shared address tokens all exceed frequency 50; `name_address_token` → no informative token intersection in both fields.
- **Category:** `address_shared_tokens_over_frequency`.

### 21. `S1-312379994` → `S2-100068944` (S2)

- **Country:** `india` ↔ `india`; compatible, no country exclusion.
- **Name:** `bombay trading private limited ↔ बॉम्बे ट्रेडिंग प्राइवेट लिमिटेड`
- **Address:** `flat number b 1 plot number 7 secotr 5 near rakhma bhawan building sanpada navi mumbai thane maharashtra ↔ flat number b 1 thane maharashtra`
- **S1 name tokens:** bombay, limited, private, trading
- **Candidate name tokens:** ट्रेडिंग, प्राइवेट, बॉम्बे, लिमिटेड
- **Shared name tokens (frequency):** —
- **S1 address tokens:** 1, 5, 7, b, bhawan, building, flat, maharashtra, mumbai, navi, near, number, plot, rakhma, sanpada, secotr, thane
- **Candidate address tokens:** 1, b, flat, maharashtra, number, thane
- **Shared address tokens (frequency):** 1=599, b=384, flat=192, maharashtra=348, number=1199, thane=73
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → no shared name token; `address_token` → shared address tokens all exceed frequency 50; `name_address_token` → no informative token intersection in both fields.
- **Category:** `address_shared_tokens_over_frequency`.

### 22. `S1-312379994` → `S2-705889282` (S2)

- **Country:** `india` ↔ `india`; compatible, no country exclusion.
- **Name:** `bombay trading private limited ↔ बॉम्बे trading private limited`
- **Address:** `flat number b 1 plot number 7 secotr 5 near rakhma bhawan building sanpada navi mumbai thane maharashtra ↔ flt number b 1 thane maharashtra`
- **S1 name tokens:** bombay, limited, private, trading
- **Candidate name tokens:** limited, private, trading, बॉम्बे
- **Shared name tokens (frequency):** limited=1751, private=1512, trading=65
- **S1 address tokens:** 1, 5, 7, b, bhawan, building, flat, maharashtra, mumbai, navi, near, number, plot, rakhma, sanpada, secotr, thane
- **Candidate address tokens:** 1, b, flt, maharashtra, number, thane
- **Shared address tokens (frequency):** 1=599, b=384, maharashtra=348, number=1199, thane=73
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → shared name tokens all exceed frequency 50; `address_token` → shared address tokens all exceed frequency 50; `name_address_token` → no informative token intersection in both fields.
- **Category:** `both_shared_token_sets_over_frequency`.

### 23. `S1-312454806` → `S2-739106848` (S2)

- **Country:** `india` ↔ `india`; compatible, no country exclusion.
- **Name:** `southern hospitality ↔ সাউদার্ন হসপিটালিটি`
- **Address:** `24 1st floor west bengal howrah kolkata anuj chamber camac street park street kolkata ↔ h no 24 1st floor kolkata howrah west bengal`
- **S1 name tokens:** hospitality, southern
- **Candidate name tokens:** সাউদার্ন, হসপিটালিটি
- **Shared name tokens (frequency):** —
- **S1 address tokens:** 1st, 24, anuj, bengal, camac, chamber, floor, howrah, kolkata, park, street, west
- **Candidate address tokens:** 1st, 24, bengal, floor, h, howrah, kolkata, no, west
- **Shared address tokens (frequency):** 1st=123, 24=69, bengal=131, floor=576, howrah=88, kolkata=191, west=455
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → no shared name token; `address_token` → shared address tokens all exceed frequency 50; `name_address_token` → no informative token intersection in both fields.
- **Category:** `address_shared_tokens_over_frequency`.

### 24. `S1-335301031` → `S2-448236984` (S2)

- **Country:** `india` ↔ `india`; compatible, no country exclusion.
- **Name:** `good consultancy limited ↔ గుడ్ కన్సల్టెన్సీ లిమిటెడ్`
- **Address:** `h no 31 a vengalrao nagar hyderabad telangana ↔ 31 a hyderabad n a telangana`
- **S1 name tokens:** consultancy, good, limited
- **Candidate name tokens:** కన్సల్టెన్సీ, గుడ్, లిమిటెడ్
- **Shared name tokens (frequency):** —
- **S1 address tokens:** 31, a, h, hyderabad, nagar, no, telangana, vengalrao
- **Candidate address tokens:** 31, a, hyderabad, n, telangana
- **Shared address tokens (frequency):** 31=53, a=502, hyderabad=176, telangana=81
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → no shared name token; `address_token` → shared address tokens all exceed frequency 50; `name_address_token` → no informative token intersection in both fields.
- **Category:** `address_shared_tokens_over_frequency`.

### 25. `S1-347000383` → `S2-389980705` (S2)

- **Country:** `india` ↔ `india`; compatible, no country exclusion.
- **Name:** `sunrise construction private limited ↔ सनराइज कंस्ट्रक्शन प्राइवेट लिमिटेड`
- **Address:** `4th floor c wing trade world kamala mills compound s b marg lower p arel mumbai mumbai city maharashtra ↔ h no d 4th floor mumbai mumbai city महाराष्ट्र`
- **S1 name tokens:** construction, limited, private, sunrise
- **Candidate name tokens:** कंस्ट्रक्शन, प्राइवेट, लिमिटेड, सनराइज
- **Shared name tokens (frequency):** —
- **S1 address tokens:** 4th, arel, b, c, city, compound, floor, kamala, lower, maharashtra, marg, mills, mumbai, p, s, trade, wing, world
- **Candidate address tokens:** 4th, city, d, floor, h, mumbai, no, महाराष्ट्र
- **Shared address tokens (frequency):** 4th=67, city=679, floor=576, mumbai=369
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → no shared name token; `address_token` → shared address tokens all exceed frequency 50; `name_address_token` → no informative token intersection in both fields.
- **Category:** `address_shared_tokens_over_frequency`.

### 26. `S1-34703985` → `S2-259476734` (S2)

- **Country:** `india` ↔ `india`; compatible, no country exclusion.
- **Name:** `gujarat consultancy private limited ↔ गुजरात कंसल्टेंसी प्राइवेट लिमिटेड`
- **Address:** `b 21 2nd floor lajpat nagar ii new delhi south delhi delhi ↔ b 21 delhi south delhi new delhi`
- **S1 name tokens:** consultancy, gujarat, limited, private
- **Candidate name tokens:** कंसल्टेंसी, गुजरात, प्राइवेट, लिमिटेड
- **Shared name tokens (frequency):** —
- **S1 address tokens:** 21, 2nd, b, delhi, floor, ii, lajpat, nagar, new, south
- **Candidate address tokens:** 21, b, delhi, new, south
- **Shared address tokens (frequency):** 21=60, b=384, delhi=497, new=668, south=227
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → no shared name token; `address_token` → shared address tokens all exceed frequency 50; `name_address_token` → no informative token intersection in both fields.
- **Category:** `address_shared_tokens_over_frequency`.

### 27. `S1-36216088` → `S2-418857877` (S2)

- **Country:** `india` ↔ `india`; compatible, no country exclusion.
- **Name:** `urban food private limited ↔ అర్బన్ ఫుడ్ ప్రైవేట్ లిమిటెడ్`
- **Address:** `flat no 504 a 5th floor plot no 1 98 4 1 13 28 and 29 image garden road jain sadguru image s capital park madhapur village serilingampally shaikpet hyderabad telangana ↔ hyderabad telangana rangareddi flat no 506 a`
- **S1 name tokens:** food, limited, private, urban
- **Candidate name tokens:** అర్బన్, ప్రైవేట్, ఫుడ్, లిమిటెడ్
- **Shared name tokens (frequency):** —
- **S1 address tokens:** 1, 13, 28, 29, 4, 504, 5th, 98, a, and, capital, flat, floor, garden, hyderabad, image, jain, madhapur, no, park, plot, road, s, sadguru, serilingampally, shaikpet, telangana, village
- **Candidate address tokens:** 506, a, flat, hyderabad, no, rangareddi, telangana
- **Shared address tokens (frequency):** a=502, flat=192, hyderabad=176, no=657, telangana=81
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → no shared name token; `address_token` → shared address tokens all exceed frequency 50; `name_address_token` → no informative token intersection in both fields.
- **Category:** `address_shared_tokens_over_frequency`.

### 28. `S1-378502980` → `S3-548747400` (S3)

- **Country:** `india` ↔ `india`; compatible, no country exclusion.
- **Name:** `unique enterprises private limited ↔ ய ன க எண டர ப ர சஸ ப ர வ ட ல ம ட ட`
- **Address:** `new no 11 old no 6 mahalingam street mahalingapuram nungambakkam chennai chennai tamil nadu ↔ new no 11 null chennai tn`
- **S1 name tokens:** enterprises, limited, private, unique
- **Candidate name tokens:** எண, க, சஸ, ட, டர, ன, ப, ம, ய, ர, ல, வ
- **Shared name tokens (frequency):** —
- **S1 address tokens:** 11, 6, chennai, mahalingam, mahalingapuram, nadu, new, no, nungambakkam, old, street, tamil
- **Candidate address tokens:** 11, chennai, new, no, null, tn
- **Shared address tokens (frequency):** 11=72, chennai=131, new=668, no=657
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → no shared name token; `address_token` → shared address tokens all exceed frequency 50; `name_address_token` → no informative token intersection in both fields.
- **Category:** `address_shared_tokens_over_frequency`.

### 29. `S1-394705639` → `S2-135005185` (S2)

- **Country:** `india` ↔ `india`; compatible, no country exclusion.
- **Name:** `shiva logistics private limited ↔ ಶಿವ ಲಾಜಿಸ್ಟಿಕ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್`
- **Address:** `neev aura level 3 19th main sector 3 hsr layout 7 a bangalore karnataka bangalore ↔ 7 a bangalore karnataka`
- **S1 name tokens:** limited, logistics, private, shiva
- **Candidate name tokens:** ಪ್ರೈವೇಟ್, ಲಾಜಿಸ್ಟಿಕ್ಸ್, ಲಿಮಿಟೆಡ್, ಶಿವ
- **Shared name tokens (frequency):** —
- **S1 address tokens:** 19th, 3, 7, a, aura, bangalore, hsr, karnataka, layout, level, main, neev, sector
- **Candidate address tokens:** 7, a, bangalore, karnataka
- **Shared address tokens (frequency):** 7=186, a=502, bangalore=226, karnataka=110
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → no shared name token; `address_token` → shared address tokens all exceed frequency 50; `name_address_token` → no informative token intersection in both fields.
- **Category:** `address_shared_tokens_over_frequency`.

### 30. `S1-408941682` → `S2-737762637` (S2)

- **Country:** `us` ↔ `us`; compatible, no country exclusion.
- **Name:** `primary care group of indianapolis llc ↔ primarycaregroup com`
- **Address:** `indianapolis in 1325 ohio street ↔ indiaanapolis 325 ohio st in`
- **S1 name tokens:** care, group, indianapolis, llc, of, primary
- **Candidate name tokens:** com, primarycaregroup
- **Shared name tokens (frequency):** —
- **S1 address tokens:** 1325, in, indianapolis, ohio, street
- **Candidate address tokens:** 325, in, indiaanapolis, ohio, st
- **Shared address tokens (frequency):** in=110, ohio=176
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → no shared name token; `address_token` → shared address tokens all exceed frequency 50; `name_address_token` → no informative token intersection in both fields.
- **Category:** `address_shared_tokens_over_frequency`.

### 31. `S1-42402348` → `S2-134131481` (S2)

- **Country:** `india` ↔ `india`; compatible, no country exclusion.
- **Name:** `smart management private limited ↔ ಸ್ಮಾರ್ಟ್ ಮ್ಯಾನೇಜ್ ಮೆಂಟ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್`
- **Address:** `flatno 201 d no 562 2 12 h i g house bangalore north bangalore karnataka ↔ h no 201 bengaluru hq region bangalore karnataka`
- **S1 name tokens:** limited, management, private, smart
- **Candidate name tokens:** ಪ್ರೈವೇಟ್, ಮೆಂಟ್, ಮ್ಯಾನೇಜ್, ಲಿಮಿಟೆಡ್, ಸ್ಮಾರ್ಟ್
- **Shared name tokens (frequency):** —
- **S1 address tokens:** 12, 2, 201, 562, bangalore, d, flatno, g, h, house, i, karnataka, no, north
- **Candidate address tokens:** 201, bangalore, bengaluru, h, hq, karnataka, no, region
- **Shared address tokens (frequency):** 201=67, bangalore=226, h=355, karnataka=110, no=657
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → no shared name token; `address_token` → shared address tokens all exceed frequency 50; `name_address_token` → no informative token intersection in both fields.
- **Category:** `address_shared_tokens_over_frequency`.

### 32. `S1-45252741` → `S3-104517610` (S3)

- **Country:** `us` ↔ `us`; compatible, no country exclusion.
- **Name:** `pediatric dental group ↔ pediatric group déntal`
- **Address:** `451 adams avenue lebanon tn ↔ <empty>`
- **S1 name tokens:** dental, group, pediatric
- **Candidate name tokens:** déntal, group, pediatric
- **Shared name tokens (frequency):** group=116, pediatric=77
- **S1 address tokens:** 451, adams, avenue, lebanon, tn
- **Candidate address tokens:** —
- **Shared address tokens (frequency):** —
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → shared name tokens all exceed frequency 50; `address_token` → no shared address token; `name_address_token` → no informative token intersection in both fields.
- **Category:** `name_shared_tokens_over_frequency`.

### 33. `S1-521683395` → `S2-193624241` (S2)

- **Country:** `india` ↔ `india`; compatible, no country exclusion.
- **Name:** `krishna tech private limited ↔ कृष्णा टेक प्राइवेट लिमिटेड`
- **Address:** `office number 1 fourth floor s number 177 1 hadapsar pune maharashtra ↔ pune maharashtra office number 1 pune`
- **S1 name tokens:** krishna, limited, private, tech
- **Candidate name tokens:** कृष्णा, टेक, प्राइवेट, लिमिटेड
- **Shared name tokens (frequency):** —
- **S1 address tokens:** 1, 177, floor, fourth, hadapsar, maharashtra, number, office, pune, s
- **Candidate address tokens:** 1, maharashtra, number, office, pune
- **Shared address tokens (frequency):** 1=599, maharashtra=348, number=1199, office=124, pune=183
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → no shared name token; `address_token` → shared address tokens all exceed frequency 50; `name_address_token` → no informative token intersection in both fields.
- **Category:** `address_shared_tokens_over_frequency`.

### 34. `S1-524269445` → `S2-217192442` (S2)

- **Country:** `india` ↔ `india`; compatible, no country exclusion.
- **Name:** `dynamic trading private limited ↔ ಡೈನಾಮಿಕ್ ಟ್ರೇಡಿಂಗ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್`
- **Address:** `21 1st cross 1st block akshaya nagar tc palya bangalore north bangalore karnataka ↔ number 0021 1st cross 1st block bengaluru rural bangalore ಕರ್ನಾಟಕ`
- **S1 name tokens:** dynamic, limited, private, trading
- **Candidate name tokens:** ಟ್ರೇಡಿಂಗ್, ಡೈನಾಮಿಕ್, ಪ್ರೈವೇಟ್, ಲಿಮಿಟೆಡ್
- **Shared name tokens (frequency):** —
- **S1 address tokens:** 1st, 21, akshaya, bangalore, block, cross, karnataka, nagar, north, palya, tc
- **Candidate address tokens:** 0021, 1st, bangalore, bengaluru, block, cross, number, rural, ಕರ್ನಾಟಕ
- **Shared address tokens (frequency):** 1st=123, bangalore=226, block=200, cross=109
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → no shared name token; `address_token` → shared address tokens all exceed frequency 50; `name_address_token` → no informative token intersection in both fields.
- **Category:** `address_shared_tokens_over_frequency`.

### 35. `S1-587336001` → `S2-905662417` (S2)

- **Country:** `us` ↔ `us`; compatible, no country exclusion.
- **Name:** `international software digital llc ↔ international sotmhwre digital llc`
- **Address:** `baytown tx 9207 franklin drive ↔ <empty>`
- **S1 name tokens:** digital, international, llc, software
- **Candidate name tokens:** digital, international, llc, sotmhwre
- **Shared name tokens (frequency):** digital=51, international=68, llc=1119
- **S1 address tokens:** 9207, baytown, drive, franklin, tx
- **Candidate address tokens:** —
- **Shared address tokens (frequency):** —
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → shared name tokens all exceed frequency 50; `address_token` → no shared address token; `name_address_token` → no informative token intersection in both fields.
- **Category:** `name_shared_tokens_over_frequency`.

### 36. `S1-59097545` → `S2-220971295` (S2)

- **Country:** `india` ↔ `india`; compatible, no country exclusion.
- **Name:** `alpha lotus consulting private limited ↔ अल्फा लोटस कंसल्टिंग प्राइवेट लिमिटेड`
- **Address:** `floor 1 world trade centre sadhu t l vaswani marg cuffe parade 58 mumbai city maharashtra mumbai ↔ 0058 maharashtra mumbai city`
- **S1 name tokens:** alpha, consulting, limited, lotus, private
- **Candidate name tokens:** अल्फा, कंसल्टिंग, प्राइवेट, लिमिटेड, लोटस
- **Shared name tokens (frequency):** —
- **S1 address tokens:** 1, 58, centre, city, cuffe, floor, l, maharashtra, marg, mumbai, parade, sadhu, t, trade, vaswani, world
- **Candidate address tokens:** 0058, city, maharashtra, mumbai
- **Shared address tokens (frequency):** city=679, maharashtra=348, mumbai=369
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → no shared name token; `address_token` → shared address tokens all exceed frequency 50; `name_address_token` → no informative token intersection in both fields.
- **Category:** `address_shared_tokens_over_frequency`.

### 37. `S1-59097545` → `S2-257682674` (S2)

- **Country:** `india` ↔ `india`; compatible, no country exclusion.
- **Name:** `alpha lotus consulting private limited ↔ अल्फा लोटस कंसल्टिंग प्राइवेट लिमिटेड`
- **Address:** `floor 1 world trade centre sadhu t l vaswani marg cuffe parade 58 mumbai city maharashtra mumbai ↔ 0058 mumbai city महाराष्ट्र`
- **S1 name tokens:** alpha, consulting, limited, lotus, private
- **Candidate name tokens:** अल्फा, कंसल्टिंग, प्राइवेट, लिमिटेड, लोटस
- **Shared name tokens (frequency):** —
- **S1 address tokens:** 1, 58, centre, city, cuffe, floor, l, maharashtra, marg, mumbai, parade, sadhu, t, trade, vaswani, world
- **Candidate address tokens:** 0058, city, mumbai, महाराष्ट्र
- **Shared address tokens (frequency):** city=679, mumbai=369
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → no shared name token; `address_token` → shared address tokens all exceed frequency 50; `name_address_token` → no informative token intersection in both fields.
- **Category:** `address_shared_tokens_over_frequency`.

### 38. `S1-599310463` → `S2-924708653` (S2)

- **Country:** `us` ↔ `us`; compatible, no country exclusion.
- **Name:** `family associates l l c ↔ family l l c associates`
- **Address:** `5715 highway 196 potwin ks ↔ <empty>`
- **S1 name tokens:** associates, c, family, l
- **Candidate name tokens:** associates, c, family, l
- **Shared name tokens (frequency):** associates=134, c=261, family=58, l=184
- **S1 address tokens:** 196, 5715, highway, ks, potwin
- **Candidate address tokens:** —
- **Shared address tokens (frequency):** —
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → shared name tokens all exceed frequency 50; `address_token` → no shared address token; `name_address_token` → no informative token intersection in both fields.
- **Category:** `name_shared_tokens_over_frequency`.

### 39. `S1-601922964` → `S3-834248782` (S3)

- **Country:** `india` ↔ `india`; compatible, no country exclusion.
- **Name:** `safa public school private limited ↔ safapublicschool com`
- **Address:** `ground floor lp 55 10 3 8 30b bejoygarh kolkata kolkata howrah west bengal ↔ ground floor howrah পশ চ মবঙ গ`
- **S1 name tokens:** limited, private, public, safa, school
- **Candidate name tokens:** com, safapublicschool
- **Shared name tokens (frequency):** —
- **S1 address tokens:** 10, 3, 30b, 55, 8, bejoygarh, bengal, floor, ground, howrah, kolkata, lp, west
- **Candidate address tokens:** floor, ground, howrah, গ, চ, পশ, মবঙ
- **Shared address tokens (frequency):** floor=576, ground=87, howrah=88
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → no shared name token; `address_token` → shared address tokens all exceed frequency 50; `name_address_token` → no informative token intersection in both fields.
- **Category:** `address_shared_tokens_over_frequency`.

### 40. `S1-717223466` → `S2-217547702` (S2)

- **Country:** `india` ↔ `india`; compatible, no country exclusion.
- **Name:** `perfect consultancy private limited ↔ পারফেক্ট কনসালটেন্সি প্রাইভেট লিমিটেড`
- **Address:** `4 1 middleton street sikkim commerce house room no 208 kolkata kolkata kolkata howrah west bengal ↔ 4 1 kolkata howrah পশ্চিমবঙ্গ`
- **S1 name tokens:** consultancy, limited, perfect, private
- **Candidate name tokens:** কনসালটেন্সি, পারফেক্ট, প্রাইভেট, লিমিটেড
- **Shared name tokens (frequency):** —
- **S1 address tokens:** 1, 208, 4, bengal, commerce, house, howrah, kolkata, middleton, no, room, sikkim, street, west
- **Candidate address tokens:** 1, 4, howrah, kolkata, পশ্চিমবঙ্গ
- **Shared address tokens (frequency):** 1=599, 4=242, howrah=88, kolkata=191
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → no shared name token; `address_token` → shared address tokens all exceed frequency 50; `name_address_token` → no informative token intersection in both fields.
- **Category:** `address_shared_tokens_over_frequency`.

### 41. `S1-72218349` → `S2-203947723` (S2)

- **Country:** `india` ↔ `india`; compatible, no country exclusion.
- **Name:** `raviya software group ↔ pyrasol`
- **Address:** `flat no 201 236 237 raviwar peth near jain mandir pune maharashtra ↔ pune flat no 201 maharashtra`
- **S1 name tokens:** group, raviya, software
- **Candidate name tokens:** pyrasol
- **Shared name tokens (frequency):** —
- **S1 address tokens:** 201, 236, 237, flat, jain, maharashtra, mandir, near, no, peth, pune, raviwar
- **Candidate address tokens:** 201, flat, maharashtra, no, pune
- **Shared address tokens (frequency):** 201=67, flat=192, maharashtra=348, no=657, pune=183
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → no shared name token; `address_token` → shared address tokens all exceed frequency 50; `name_address_token` → no informative token intersection in both fields.
- **Category:** `address_shared_tokens_over_frequency`.

### 42. `S1-72218349` → `S2-691452538` (S2)

- **Country:** `india` ↔ `india`; compatible, no country exclusion.
- **Name:** `raviya software group ↔ lyraquo`
- **Address:** `flat no 201 236 237 raviwar peth near jain mandir pune maharashtra ↔ 201 maharashtra pune`
- **S1 name tokens:** group, raviya, software
- **Candidate name tokens:** lyraquo
- **Shared name tokens (frequency):** —
- **S1 address tokens:** 201, 236, 237, flat, jain, maharashtra, mandir, near, no, peth, pune, raviwar
- **Candidate address tokens:** 201, maharashtra, pune
- **Shared address tokens (frequency):** 201=67, maharashtra=348, pune=183
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → no shared name token; `address_token` → shared address tokens all exceed frequency 50; `name_address_token` → no informative token intersection in both fields.
- **Category:** `address_shared_tokens_over_frequency`.

### 43. `S1-728011176` → `S3-949643807` (S3)

- **Country:** `us` ↔ `us`; compatible, no country exclusion.
- **Name:** `choice resorts incorporated ↔ choice résorts incorporated incorporated`
- **Address:** `711 pleasant lane lagrange in ↔ <empty>`
- **S1 name tokens:** choice, incorporated, resorts
- **Candidate name tokens:** choice, incorporated, résorts
- **Shared name tokens (frequency):** choice=55, incorporated=766
- **S1 address tokens:** 711, in, lagrange, lane, pleasant
- **Candidate address tokens:** —
- **Shared address tokens (frequency):** —
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → shared name tokens all exceed frequency 50; `address_token` → no shared address token; `name_address_token` → no informative token intersection in both fields.
- **Category:** `name_shared_tokens_over_frequency`.

### 44. `S1-785596927` → `S3-110844528` (S3)

- **Country:** `india` ↔ `india`; compatible, no country exclusion.
- **Name:** `ganpati and company ↔ smt gsanaptit and company 56067`
- **Address:** `level 5 dnyanvatsal complex opp vandevi mandir karve nagar pune maharashtra ↔ h no 546 lecel 5 pune मह र ष ट र`
- **S1 name tokens:** and, company, ganpati
- **Candidate name tokens:** 56067, and, company, gsanaptit, smt
- **Shared name tokens (frequency):** and=592, company=232
- **S1 address tokens:** 5, complex, dnyanvatsal, karve, level, maharashtra, mandir, nagar, opp, pune, vandevi
- **Candidate address tokens:** 5, 546, h, lecel, no, pune, ट, मह, र, ष
- **Shared address tokens (frequency):** 5=190, pune=183
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → shared name tokens all exceed frequency 50; `address_token` → shared address tokens all exceed frequency 50; `name_address_token` → no informative token intersection in both fields.
- **Category:** `both_shared_token_sets_over_frequency`.

### 45. `S1-83402127` → `S2-272290151` (S2)

- **Country:** `india` ↔ `india`; compatible, no country exclusion.
- **Name:** `raj global limited ↔ राज ग्लोबल लिमिटेड`
- **Address:** `611 a tulsiani chambers nariman point mumbai mumbai city maharashtra ↔ महाराष्ट्र mumbai mumbai city 61 a`
- **S1 name tokens:** global, limited, raj
- **Candidate name tokens:** ग्लोबल, राज, लिमिटेड
- **Shared name tokens (frequency):** —
- **S1 address tokens:** 611, a, chambers, city, maharashtra, mumbai, nariman, point, tulsiani
- **Candidate address tokens:** 61, a, city, mumbai, महाराष्ट्र
- **Shared address tokens (frequency):** a=502, city=679, mumbai=369
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → no shared name token; `address_token` → shared address tokens all exceed frequency 50; `name_address_token` → no informative token intersection in both fields.
- **Category:** `address_shared_tokens_over_frequency`.

### 46. `S1-875918918` → `S2-965053021` (S2)

- **Country:** `us` ↔ `us`; compatible, no country exclusion.
- **Name:** `dental clinic llc ↔ dental llc services`
- **Address:** `881 still willow lane wendell nc ↔ <empty>`
- **S1 name tokens:** clinic, dental, llc
- **Candidate name tokens:** dental, llc, services
- **Shared name tokens (frequency):** dental=55, llc=1119
- **S1 address tokens:** 881, lane, nc, still, wendell, willow
- **Candidate address tokens:** —
- **Shared address tokens (frequency):** —
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → shared name tokens all exceed frequency 50; `address_token` → no shared address token; `name_address_token` → no informative token intersection in both fields.
- **Category:** `name_shared_tokens_over_frequency`.

### 47. `S1-875918918` → `S3-967195838` (S3)

- **Country:** `us` ↔ `us`; compatible, no country exclusion.
- **Name:** `dental clinic llc ↔ dental clinic associates`
- **Address:** `881 still willow lane wendell nc ↔ <empty>`
- **S1 name tokens:** clinic, dental, llc
- **Candidate name tokens:** associates, clinic, dental
- **Shared name tokens (frequency):** clinic=105, dental=55
- **S1 address tokens:** 881, lane, nc, still, wendell, willow
- **Candidate address tokens:** —
- **Shared address tokens (frequency):** —
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → shared name tokens all exceed frequency 50; `address_token` → no shared address token; `name_address_token` → no informative token intersection in both fields.
- **Category:** `name_shared_tokens_over_frequency`.

### 48. `S1-946780862` → `S3-542004037` (S3)

- **Country:** `india` ↔ `india`; compatible, no country exclusion.
- **Name:** `almighty technologys india private limited ↔ tialmighty com`
- **Address:** `a 4 second portion pushpanjali farm road village vijwasan new delhi south west delhi delhi ↔ a 4 null south west delhi new delhi द ल ल`
- **S1 name tokens:** almighty, india, limited, private, technologys
- **Candidate name tokens:** com, tialmighty
- **Shared name tokens (frequency):** —
- **S1 address tokens:** 4, a, delhi, farm, new, portion, pushpanjali, road, second, south, vijwasan, village, west
- **Candidate address tokens:** 4, a, delhi, new, null, south, west, द, ल
- **Shared address tokens (frequency):** 4=242, a=502, delhi=497, new=668, south=227, west=455
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → no shared name token; `address_token` → shared address tokens all exceed frequency 50; `name_address_token` → no informative token intersection in both fields.
- **Category:** `address_shared_tokens_over_frequency`.

### 49. `S1-949674979` → `S3-670907870` (S3)

- **Country:** `india` ↔ `india`; compatible, no country exclusion.
- **Name:** `my logistics private limited ↔ म य ल ज स ट क स प र इव ट ल म ट ड`
- **Address:** `a 401 lakshachandi apts krishna vatika marg gokuldham gor egaon east mumbai maharashtra ↔ a 40 mumbai mh`
- **S1 name tokens:** limited, logistics, my, private
- **Candidate name tokens:** इव, क, ज, ट, ड, प, म, य, र, ल, स
- **Shared name tokens (frequency):** —
- **S1 address tokens:** 401, a, apts, east, egaon, gokuldham, gor, krishna, lakshachandi, maharashtra, marg, mumbai, vatika
- **Candidate address tokens:** 40, a, mh, mumbai
- **Shared address tokens (frequency):** a=502, mumbai=369
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → no shared name token; `address_token` → shared address tokens all exceed frequency 50; `name_address_token` → no informative token intersection in both fields.
- **Category:** `address_shared_tokens_over_frequency`.

### 50. `S1-967751721` → `S2-375000485` (S2)

- **Country:** `india` ↔ `india`; compatible, no country exclusion.
- **Name:** `jain logistics limited ↔ जैन लॉजिस्टिक्स लिमिटेड`
- **Address:** `shop no 13 survey no 690 1 purandhar co op housing society maharshi nagar pune maharashtra ↔ shop no 13 pune poona maharashtra`
- **S1 name tokens:** jain, limited, logistics
- **Candidate name tokens:** जैन, लिमिटेड, लॉजिस्टिक्स
- **Shared name tokens (frequency):** —
- **S1 address tokens:** 1, 13, 690, co, housing, maharashtra, maharshi, nagar, no, op, pune, purandhar, shop, society, survey
- **Candidate address tokens:** 13, maharashtra, no, poona, pune, shop
- **Shared address tokens (frequency):** 13=79, maharashtra=348, no=657, pune=183, shop=93
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → no shared name token; `address_token` → shared address tokens all exceed frequency 50; `name_address_token` → no informative token intersection in both fields.
- **Category:** `address_shared_tokens_over_frequency`.

### 51. `S1-997731506` → `S2-208204396` (S2)

- **Country:** `india` ↔ `india`; compatible, no country exclusion.
- **Name:** `ace tech tech private limited ↔ एस टेक टेक प्राइवेट लिमिटेड`
- **Address:** `8 nalanda appartments d block vikas puri new delhi west delhi delhi ↔ 8 new delhi west delhi delhi`
- **S1 name tokens:** ace, limited, private, tech
- **Candidate name tokens:** एस, टेक, प्राइवेट, लिमिटेड
- **Shared name tokens (frequency):** —
- **S1 address tokens:** 8, appartments, block, d, delhi, nalanda, new, puri, vikas, west
- **Candidate address tokens:** 8, delhi, new, west
- **Shared address tokens (frequency):** 8=177, delhi=497, new=668, west=455
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → no shared name token; `address_token` → shared address tokens all exceed frequency 50; `name_address_token` → no informative token intersection in both fields.
- **Category:** `address_shared_tokens_over_frequency`.

### 52. `S1-997731506` → `S2-749005505` (S2)

- **Country:** `india` ↔ `india`; compatible, no country exclusion.
- **Name:** `ace tech tech private limited ↔ एस टेक टेक प्राइवेट लिमिटेड`
- **Address:** `8 nalanda appartments d block vikas puri new delhi west delhi delhi ↔ 8 new delhi west delhi delhi`
- **S1 name tokens:** ace, limited, private, tech
- **Candidate name tokens:** एस, टेक, प्राइवेट, लिमिटेड
- **Shared name tokens (frequency):** —
- **S1 address tokens:** 8, appartments, block, d, delhi, nalanda, new, puri, vikas, west
- **Candidate address tokens:** 8, delhi, new, west
- **Shared address tokens (frequency):** 8=177, delhi=497, new=668, west=455
- **Exact-key diagnostics:** exact name = `False`; exact address = `False`; exact name bucket = `0`; exact address bucket = `0`.
- **Default exclusion:** `country_name_exact` → no exact normalized name; `country_address_exact` → no exact normalized address; `name_token` → no shared name token; `address_token` → shared address tokens all exceed frequency 50; `name_address_token` → no informative token intersection in both fields.
- **Category:** `address_shared_tokens_over_frequency`.

## Recommendation

Do not change the default blocking configuration as part of this investigation.
The current 99.50% recall is achieved with 161,162 candidates (about 0.51% of the
31,332,000-pair Cartesian space). The misses are concentrated in frequent address/name
tokens and are therefore precisely the cases most likely to create false-positive
blocks. If recall must be improved, benchmark the targeted combination of a bounded
multi-address-token rule plus a modest name-token cap against downstream pair scoring
and entity-level macro F0.5; do not enable it solely from blocking recall. Keep
country-independent and unrestricted fuzzy rules out of the default path unless a
later validation split demonstrates a material end-to-end gain.

## Reproduction

```bash
myenv/bin/python -m pytest -q
myenv/bin/python scripts/evaluate_blocking.py
```

All results in this document are from the sample files and the current default
configuration on 2026-09-25. No default code or configuration was changed for this
investigation.
