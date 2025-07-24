import nltk
nltk.data.path.append('/Users/m2/Downloads/ChatBot Tool/venv/nltk_data/')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- 1. FAQs Collect Karna ---
# Yahan apni FAQs add karo (Question: Answer format mein)
# Aap jitne chahe utne FAQs add kar sakte hain.
faq_data = {
    "hello": "Hi there! How can I help you today?",
    "hi": "Hello! How may I assist you?",
    "what is your name": "I am a simple FAQ Chatbot, created to help you.",
    "how are you": "I'm doing great, thank you for asking! How about you?",
    "what are your services": "I can answer frequently asked questions about any topic you provide me with. Just ask!",
    "do you offer refunds": "Yes, we offer a 30-day money-back guarantee on all our products. Please check our refund policy page for details.",
    "how to reset my password": "You can reset your password by going to the 'Forgot Password' link on the login page. An email with reset instructions will be sent to your registered email ID.",
    "where is your office located": "We are an online-only service and do not have a physical office location at the moment.",
    "what is your pricing": "Our pricing varies based on the features. Please visit our 'Pricing' page for detailed information on different plans.",
    "customer support number": "For customer support, please call us at +91-9876543210 during business hours.",
    "how can I contact support": "You can contact our support team via email at support@example.com or call us at +91-9876543210.",
    "what is your business hours": "Our support team is available from 9 AM to 6 PM IST, Monday to Friday.",
    "bye": "Goodbye! Have a great day!",
    "thank you": "You're welcome! Let me know if you have more questions."
}

# --- 2. Text Preprocessing Functions ---
lemmatizer = WordNetLemmatizer()
# NLTK stopwords English ke liye
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower() # 1. Lowercasing
    text = re.sub(r'[^a-z0-9\s]', '', text) # 2. Remove punctuation and special characters
    words = nltk.word_tokenize(text) # 3. Tokenization (text ko words mein todna)
    words = [word for word in words if word not in stop_words] # 4. Remove stopwords
    words = [lemmatizer.lemmatize(word) for word in words] # 5. Lemmatization (words ko root form mein laana)
    return " ".join(words) # Cleaned words ko wapas string mein join karna

# Saare FAQ questions ko preprocess karna, matching ke liye
faq_questions_processed = [preprocess_text(q) for q in faq_data.keys()]

# --- 3. TF-IDF Vectorizer Setup ---
# TfidfVectorizer ko FAQ questions par train karna
vectorizer = TfidfVectorizer().fit(faq_questions_processed)
faq_vectors = vectorizer.transform(faq_questions_processed)


# --- 4. Chatbot Logic ---
def get_response(user_query):
    processed_query = preprocess_text(user_query)

    # Agar user ne kuch nahi likha toh
    if not processed_query.strip():
        return "I didn't understand that. Please type something."

    # User query ko TF-IDF vector mein convert karna
    query_vector = vectorizer.transform([processed_query])

    # Cosine Similarity calculate karna
    # Yeh dekhta hai ki user ki query kis FAQ question se sabse zyada milti julti hai.
    similarities = cosine_similarity(query_vector, faq_vectors)

    # Sabse similar FAQ ka index dhundhna
    best_match_index = similarities.argmax()

    # Best match ki similarity score
    similarity_score = similarities[0, best_match_index]

    # Ek threshold set karna. Agar similarity isse kam hai, toh general response denge.
    # Aap is value ko adjust kar sakte hain (e.g., 0.5, 0.6, 0.7)
    similarity_threshold = 0.5

    if similarity_score > similarity_threshold:
        # Agar similarity threshold se zyada hai, toh matched FAQ ka answer do
        matched_question = list(faq_data.keys())[best_match_index]
        return faq_data[matched_question]
    else:
        # Agar similarity kam hai, toh general "I don't know" response do
        return "I'm sorry, I don't have an answer for that specific question. Could you try asking something else or rephrasing your question?"

# --- 5. Simple Chatbot UI (Terminal-based) ---
print("Hello! I am your friendly FAQ Chatbot. Ask me anything about our services or general queries.")
print("Type 'quit' or 'exit' to end the conversation.")
print("-" * 60) # Ek line separation ke liye

while True:
    user_input = input("You: ")
    if user_input.lower() in ['quit', 'exit']:
        print("Chatbot: Goodbye! Have a great day!")
        break

    response = get_response(user_input)
    print(f"Chatbot: {response}")
    print("-" * 60) # Har response ke baad separation