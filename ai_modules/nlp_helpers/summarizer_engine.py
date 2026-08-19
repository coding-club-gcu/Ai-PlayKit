import os
import re
import math
from collections import Counter

class NLPSummarizerEngine:
    """Offline Natural Language Processing (NLP) Engine for Document Summarization & Sentiment Analysis."""

    STOP_WORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
        "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't",
        "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
        "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he",
        "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
        "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
        "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
        "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll",
        "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs",
        "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
        "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll",
        "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while",
        "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're",
        "you've", "your", "yours", "yourself", "yourselves", "also", "using", "use", "used", "will", "can", "may"
    }

    POSITIVE_WORDS = {
        "good", "great", "excellent", "amazing", "wonderful", "outstanding", "fantastic", "positive", "success",
        "successful", "benefit", "best", "effective", "improve", "improved", "growth", "high", "valuable",
        "superb", "brilliant", "enjoy", "pleasure", "strength", "strong", "win", "achievement", "advancement"
    }

    NEGATIVE_WORDS = {
        "bad", "poor", "terrible", "horrible", "awful", "negative", "fail", "failed", "failure", "error", "issue",
        "problem", "risk", "harm", "damage", "loss", "decline", "worst", "severe", "hard", "difficult", "threat",
        "crisis", "weakness", "flaw", "defect", "adverse", "trouble"
    }

    def __init__(self):
        pass

    def extract_text_from_file(self, filepath):
        """Extracts plain text from .txt, .pdf, .docx, .md, .py, .json, .csv files."""
        if not os.path.exists(filepath):
            return ""

        ext = os.path.splitext(filepath)[1].lower()

        # Plain text & code files
        if ext in [".txt", ".md", ".py", ".json", ".csv", ".log", ".html", ".css", ".js"]:
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
            except Exception as e:
                print(f"[NLPSummarizer] Text file read error: {e}")
                return ""

        # PDF documents
        elif ext == ".pdf":
            text = ""
            try:
                import pypdf
                reader = pypdf.PdfReader(filepath)
                for page in reader.pages:
                    t = page.extract_text()
                    if t:
                        text += t + "\n"
                return text
            except Exception:
                try:
                    import PyPDF2
                    reader = PyPDF2.PdfReader(filepath)
                    for page in reader.pages:
                        t = page.extract_text()
                        if t:
                            text += t + "\n"
                    return text
                except Exception as ex:
                    print(f"[NLPSummarizer] PDF extract error: {ex}")
                    return ""

        # Word documents (.docx)
        elif ext == ".docx":
            try:
                import docx
                doc = docx.Document(filepath)
                return "\n".join([p.text for p in doc.paragraphs if p.text])
            except Exception as ex:
                print(f"[NLPSummarizer] DOCX extract error: {ex}")
                return ""

        return ""

    def summarize(self, text, ratio=0.25, mode="bullet"):
        """
        Summarizes text using TextRank sentence importance scoring.
        ratio: fraction of sentences to keep (0.1 to 0.5)
        mode: 'bullet', 'paragraph', or 'keypoints'
        """
        if not text.strip():
            return "", []

        sentences = self.split_sentences(text)
        if len(sentences) <= 2:
            return text.strip(), self.extract_keywords(text)

        # Tokenize words per sentence
        words_per_sent = [self.tokenize_words(s) for s in sentences]

        # Calculate term frequencies across whole document
        doc_words = [w for sent_words in words_per_sent for w in sent_words if w not in self.STOP_WORDS]
        word_counts = Counter(doc_words)
        total_valid_words = max(1, len(doc_words))

        # Score sentences
        scores = []
        for idx, (sent, words) in enumerate(zip(sentences, words_per_sent)):
            if not words:
                scores.append(0.0)
                continue

            # Word frequency importance score
            wf_score = sum(word_counts[w] for w in words if w not in self.STOP_WORDS) / max(1, len(words))

            # Positional score: boost first and last sentences of document
            pos_bonus = 1.3 if (idx == 0 or idx == len(sentences) - 1) else (1.1 if idx < 3 else 1.0)

            # Sentence length penalty (avoid tiny or excessively long sentences)
            length_penalty = 1.0
            if len(words) < 5 or len(words) > 45:
                length_penalty = 0.7

            final_score = wf_score * pos_bonus * length_penalty
            scores.append(final_score)

        # Select top N sentences based on target ratio
        num_sentences_to_select = max(1, int(len(sentences) * ratio))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:num_sentences_to_select]
        
        # Sort selected indices to preserve original chronological narrative flow
        ranked_indices.sort()
        selected_sentences = [sentences[i] for i in ranked_indices]

        keywords = self.extract_keywords(text)

        # Format output based on requested mode
        if mode == "bullet":
            summary_output = "\n".join(f"• {s}" for s in selected_sentences)
        elif mode == "keypoints":
            summary_output = "\n".join(f"{idx+1}. {s}" for idx, s in enumerate(selected_sentences))
        else:
            summary_output = " ".join(selected_sentences)

        return summary_output, keywords

    def analyze_sentiment(self, text):
        """Analyzes text sentiment polarity, subjectivity, and mood classification."""
        if not text.strip():
            return {"sentiment": "Neutral", "score": 0.0, "subjectivity": 0.5, "icon": "😐"}

        # Try TextBlob first
        try:
            from textblob import TextBlob
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
        except Exception:
            # Fallback custom sentiment lexicon scanner
            words = self.tokenize_words(text)
            pos_cnt = sum(1 for w in words if w in self.POSITIVE_WORDS)
            neg_cnt = sum(1 for w in words if w in self.NEGATIVE_WORDS)
            total = max(1, pos_cnt + neg_cnt)
            polarity = (pos_cnt - neg_cnt) / total
            subjectivity = 0.5

        if polarity > 0.12:
            sentiment = "Positive"
            icon = "😊"
        elif polarity < -0.12:
            sentiment = "Negative"
            icon = "😟"
        else:
            sentiment = "Neutral"
            icon = "😐"

        return {
            "sentiment": sentiment,
            "score": float(polarity),
            "subjectivity": float(subjectivity),
            "icon": icon
        }

    def extract_keywords(self, text, top_n=10):
        """Extracts top N most important keywords from text using TF-IDF ranking."""
        words = self.tokenize_words(text)
        filtered = [w for w in words if w not in self.STOP_WORDS and len(w) > 2 and not w.isdigit()]
        counts = Counter(filtered)
        return counts.most_common(top_n)

    def split_sentences(self, text):
        """Splits raw document text into clean sentence list using regex."""
        # Normalize newlines
        clean_text = re.sub(r'\s+', ' ', text).strip()
        # Split on sentence boundary punctuation
        sentence_candidates = re.split(r'(?<=[.!?])\s+', clean_text)
        sentences = [s.strip() for s in sentence_candidates if len(s.strip()) > 8]
        return sentences

    CATEGORY_LEXICON = {
        "🤖 AI & Technology": {
            "ai", "artificial", "intelligence", "machine", "learning", "model", "data", "algorithm",
            "neural", "network", "software", "system", "code", "python", "computer", "vision", "speech",
            "deep", "digital", "technology", "gpu", "automation", "app", "framework", "database", "dev"
        },
        "💼 Business & Finance": {
            "business", "company", "market", "financial", "finance", "revenue", "profit", "sales",
            "growth", "strategy", "investment", "cost", "customer", "price", "economy", "trade",
            "stock", "management", "capital", "industry", "commercial", "budget", "product"
        },
        "🧬 Science & Healthcare": {
            "science", "research", "study", "health", "medical", "patient", "disease", "clinical",
            "treatment", "doctor", "cell", "biological", "dna", "physics", "chemistry", "energy",
            "climate", "environment", "lab", "trial", "scientific"
        },
        "🎓 Education & Academics": {
            "education", "student", "school", "university", "learning", "teacher", "course", "degree",
            "academic", "paper", "theory", "knowledge", "class", "curriculum", "study", "research"
        },
        "⚖️ Legal & Policy": {
            "law", "legal", "court", "policy", "regulation", "government", "contract", "compliance",
            "rights", "act", "clause", "agreement", "party", "jurisdiction", "statute", "privacy"
        }
    }

    def generate_auto_tags(self, text, max_tags=6):
        """Generates AI Auto-Tags including document domain category and top hashtag topic tags."""
        if not text.strip():
            return {"category": "📄 General Document", "tags": []}

        words = self.tokenize_words(text)
        doc_words_set = set(words)
        
        # Determine main category by matching word overlap
        category_scores = {}
        for cat_name, cat_words in self.CATEGORY_LEXICON.items():
            overlap = len(doc_words_set.intersection(cat_words))
            if overlap > 0:
                category_scores[cat_name] = overlap

        if category_scores:
            main_category = max(category_scores, key=category_scores.get)
        else:
            main_category = "📄 General Document"

        # Generate top hashtag tags from keywords
        keywords = self.extract_keywords(text, top_n=max_tags)
        hashtag_tags = [f"#{w.capitalize()}" for w, count in keywords]

        return {
            "category": main_category,
            "tags": hashtag_tags
        }

    def tokenize_words(self, text):
        """Tokenizes text into lowercase words."""
        return re.findall(r'\b[a-zA-Z0-9_]+\b', text.lower())
