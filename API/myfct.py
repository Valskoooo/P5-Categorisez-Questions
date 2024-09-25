import spacy
import re

# Fonction pour le pre-processing du texte
# Charger le modèle de langue
nlp = spacy.load("en_core_web_sm")
# Liste des mots spécifiques à conserver (comme "C++", "C#", etc.)
whitelist = ["c#", "c++"]
# Fonction pour le traitement du texte
def preprocess_text(text):
    # Enlever les balises HTML
    #soup = BeautifulSoup(text, "lxml").get_text()
    doc = nlp(text)  # Traiter le texte avec spaCy
    
    # Expression régulière pour vérifier les caractères anglais uniquement (ASCII)
    def is_english(token):
        return re.match(r'^[a-zA-Z0-9+.#]+$', token)

    tokens = [
        # Utiliser le lemme sauf si le mot est dans la whitelist
        token.lemma_.lower() if token.lemma_.lower() not in whitelist else token.text.lower()
        for token in doc 
        if not token.is_stop                         # Ne pas inclure les stopwords
        and not token.is_punct                       # Ne pas inclure la ponctuation
        and not token.like_num                       # Ne pas inclure les chiffres
        and len(token.lemma_) >= 3                    # Exclure les tokens trop courts
        and (is_english(token.text) or token.text.lower() in whitelist)  # Garder les mots anglais ou ceux de la whitelist
    ]
    print(tokens)
    return tokens

def filter_words_in_vocab(words, w2v_model):
    """Filtrer les mots pour ne conserver que ceux présents dans le vocabulaire du modèle Word2Vec"""
    return [word for word in words if word in w2v_model.wv]