# The Effect of Natural Language Morphology on the Structure of NLP Systems

> A cross-lingual study exploring how grammatical structure across typologically distinct language families influences the internal representations of language-specific BERT models.

**Course:** BIL471 — Natural Language Processing  
**Author:** Jouman Hmeid (211410033)  
**Institution:** TOBB University of Economics and Technology

---

## Research Question

> *With different transformer models designed and specialised for different languages, what can we learn about how linguistic features — and specifically morphological differences between languages from three widely distinct language families — impact the tuning and internal structure of NLP models?*

The three languages chosen represent maximally different typological profiles:

| Language | Family | Morphological Type | Word Order |
|---|---|---|---|
| English | Indo-European (Germanic) | Analytic (minimal inflection) | SVO |
| Turkish | Turkic | Agglutinative (rich suffixation) | SOV |
| Mandarin Chinese | Sino-Tibetan | Isolating (no inflection) | SVO |

This contrast was intentional: Turkish forms entire phrases through suffix stacking on a single root (e.g. *çalışmalarını* = work+PLURAL+POSSESSIVE+ACCUSATIVE), while Mandarin expresses the same relationships through word order and separate particles with no morphological change to the root. English sits between the two. These differences have direct consequences for how tokenisers break text apart and how attention heads learn to distribute focus.

---

## Methodology

The project used three complementary analysis methods, progressing from surface-level grammatical tagging to deep internal model analysis.

### Method 1 — POS Tagging Comparison

**Goal:** Establish baseline grammatical profiles for each language and observe how different NLP tools handle each language's morphological complexity.

**Tools used:**
- `spaCy` (`en_core_web_sm`, `zh_core_web_sm`) — for English and Chinese
- `Stanza` (English, Turkish, Chinese pipelines) — Universal Dependencies-aligned tagger
- `HanLP` (via REST API) — specialised Chinese NLP toolkit
- `dbmdz/bert-base-turkish-uncased` via HuggingFace pipeline — for Turkish NER/tagging

**Key observation:** Turkish was not available in spaCy at the time of the study, illustrating a broader point — agglutinative languages are underrepresented in mainstream NLP tooling. Mandarin presented a different challenge: lemmatization is largely irrelevant in Chinese since words do not inflect, meaning a key preprocessing step used in English and Turkish pipelines simply does not apply.

**Sample POS output comparison (equivalent sentence: "I love natural language processing"):**

```
English (spaCy):          Chinese (spaCy):
I        PRON             我      PRON  PN
love     VERB             喜欢    VERB  VV
natural  ADJ              学习    VERB  VV
language NOUN             自然    NOUN  NN
processing NOUN           语言    NOUN  NN
                          处理    VERB  VV
```

Note: what is a single noun phrase in English ("natural language processing") maps to three separate content words in Chinese, each carrying its own POS tag. This has direct downstream effects on tokenisation and sequence length in BERT models.

---

### Method 2 — Dependency Tree Analysis

**Goal:** Visualise and compare the syntactic structure of equivalent sentences across the three languages, focusing on how grammatical relations (subject, object, modifier, etc.) are encoded structurally.

**Tools used:**
- `UDPipe` (Universal Dependencies models for English, Turkish, Chinese) — tokenise, tag, parse, lemmatize
- `Stanza` — alternative dependency parsing with visualisation
- `HanLP` REST API — multilingual dependency parsing including `.pretty_print()` output
- `graphviz` — custom dependency tree visualisation

**Sentence types tested:** Simple declarative, conditional, passive voice, interrogative — all expressed equivalently in each of the three languages.

**Key observations:**

- **Turkish** dependency trees are noticeably deeper and more right-branching. Because suffixes encode grammatical roles, the head word typically appears at the end of the clause, and multiple dependents stack on a single root through suffix attachment rather than through separate function words.
- **Mandarin Chinese** trees are flatter. Grammatical relationships that require separate nodes in English or Turkish (e.g. prepositions, plural markers) are either absent or implied through context and word order alone.
- **English** trees are moderate in depth, with determiners and prepositions creating additional nodes that do not appear in Mandarin.

This structural difference in dependency trees is significant for transformer models: a Turkish sentence conveying the same meaning as an English one will typically produce a longer token sequence (due to suffix-heavy words being split into multiple subword tokens) and different head-word positioning, which directly affects where attention is concentrated.

---

### Method 3 — BERT Attention Head Analysis & Probing Classifiers

**Goal:** Examine the internal representations of language-specific BERT models to determine how well each model encodes syntactic structure, and whether attention patterns reflect the grammatical properties of each language.

**Models used:**
- `bert-base-uncased` — English
- `dbmdz/bert-base-turkish-uncased` — Turkish
- `bert-base-chinese` — Mandarin Chinese

#### Part A: Attention Visualisation

Using `bertviz`, the multi-head attention weights from each model were extracted and visualised across all 12 layers and 12 heads.

```python
from bertviz import head_view

outputs = model(**inputs)
attention = outputs.attentions  # (num_layers, batch_size, num_heads, seq_len, seq_len)
head_view(attention, tokens)
```

**Key observations:**
- In **Turkish**, attention patterns at layer 3 show noticeably wider distribution across tokens compared to English. This is consistent with the agglutinative structure: the model must attend across longer token sequences to resolve grammatical roles that in English would be signalled by a single short function word.
- In **Mandarin**, attention is often concentrated on the immediately adjacent token, reflecting the analytic, order-dependent nature of the language.
- **English** shows intermediate patterns with some heads clearly specialising in subject-verb relationships and determiner-noun links.

The computation graph for the Chinese model was also visualised using `torchviz` to inspect the forward-pass structure.

#### Part B: Probing Classifiers

To quantify how well syntactic information (specifically Part-of-Speech categories) is encoded in each model's hidden states, logistic regression probing classifiers were trained on the token representations extracted from layer 3.

```python
# Extract layer 3 hidden states
layer_output = hidden_states[3]           # (batch, seq_len, 768)
token_features = layer_output[0, 1:-1, :] # exclude [CLS] and [SEP]

# Train probe
from sklearn.linear_model import LogisticRegression
probe = LogisticRegression(max_iter=1000)
probe.fit(token_features_np, pos_labels)

# Evaluate
accuracy = accuracy_score(pos_labels, probe.predict(token_features_np))
```

This was run for English (`bert-base-uncased`) and Chinese (`bert-base-chinese`) on matched sentences with manually assigned POS labels. The accuracy of the probing classifier serves as a measure of how linearly separable POS information is in the model's representations at that layer — a well-known interpretability technique in NLP research.

**Summary of findings:**

| Model | Language | Morphological Type | Observation |
|---|---|---|---|
| bert-base-uncased | English | Analytic | Attention heads show clear syntactic specialisation; probing accuracy high for core POS categories |
| dbmdz/bert-base-turkish-uncased | Turkish | Agglutinative | Longer token sequences due to subword splitting of suffixed forms; attention distributed more broadly |
| bert-base-chinese | Mandarin | Isolating | Denser, more local attention patterns; POS distinctions partially collapsed (e.g. no morphological verb/noun distinction) |

---

## Repository Structure

```
Natural_Language_Experimental_Research_Project/
│
├── 471part1_2.ipynb          # Method 1 (POS tagging) + Method 3 (BERT attention & probing)
├── 471part2.ipynb            # Method 2 (dependency trees) + additional BERT embedding analysis
├── 471projebert.ipynb        # Standalone BERT analysis notebook (attention + CLS embeddings)
├── 471_sunum_211410033.pptx  # Project presentation slides
└── README.md
```

---

## Dependencies

```bash
pip install torch transformers bertviz
pip install spacy stanza
pip install scikit-learn numpy
pip install graphviz torchviz
pip install hanlp hanlp-restful
pip install ufal.udpipe

# spaCy language models
python -m spacy download en_core_web_sm
python -m spacy download zh_core_web_sm

# Stanza language models (downloaded at runtime)
import stanza
stanza.download('en')
stanza.download('tr')
stanza.download('zh')
```

> **Note:** UDPipe requires separate language model files (`.udpipe` format) downloaded from the [UDPipe model repository](https://ufal.mff.cuni.cz/udpipe/1/models). The notebooks reference local model paths which you will need to update.

---

## Key Concepts

**Agglutinative morphology** — a word formation strategy where grammatical meaning (tense, case, number, possession) is expressed by chaining suffixes onto a root. Turkish is a canonical example. This matters for NLP because a single Turkish word often tokenises into many subword pieces, producing longer sequences and requiring the model to track dependencies across more tokens.

**Isolating morphology** — words do not change form; grammatical relationships are expressed through word order and separate particles. Mandarin Chinese is the primary example here. Lemmatization is not meaningful for such languages, and many POS distinctions present in inflected languages collapse.

**Probing classifiers** — a standard interpretability technique where a simple classifier (here, logistic regression) is trained on the frozen hidden states of a pretrained model to determine what information is encoded at each layer. High accuracy means the information is linearly present in the representations; low accuracy suggests it is not encoded at that layer or is encoded in a non-linear way.

**BERTviz** — an open-source tool for visualising multi-head attention weights in transformer models, allowing inspection of which tokens each attention head focuses on at each layer.

---

## References

- Devlin et al. (2019) — *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*
- Tenney et al. (2019) — *BERT Rediscovers the Classical NLP Pipeline*
- Clark et al. (2019) — *What Does BERT Look at? An Analysis of BERT's Attention*
- Vania & Lopez (2017) — *From Characters to Words to in Between: Do We Capture Morphology?*
- [dbmdz/bert-base-turkish-uncased](https://huggingface.co/dbmdz/bert-base-turkish-uncased) — HuggingFace
- [UDPipe](https://ufal.mff.cuni.cz/udpipe) — Institute of Formal and Applied Linguistics, Prague
- [Stanza](https://stanfordnlp.github.io/stanza/) — Stanford NLP Group
