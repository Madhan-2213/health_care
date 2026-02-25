import re
import uuid
import os
from difflib import SequenceMatcher
from collections import Counter

import mysql.connector

try:
    import spacy
except ImportError:
    spacy = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def load_nlp_pipeline():
    if spacy is None:
        return None
    try:
        return spacy.load("en_core_web_md")
    except Exception:
        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        return nlp


nlp = load_nlp_pipeline()

db_config = {
    "host": "localhost",
    "user": "root",
    "password": "mayura_ziani",
    "database": "health_care_chatbot"
}

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or OpenAI is None:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None


def build_genai_messages(user_text, best_row, confidence, history):
    if best_row is None:
        context_block = "No trusted protocol match was found. Ask for clarification and request key terms."
    else:
        context_block = (
            f"Matched Protocol Question: {best_row['question']}\n"
            f"Matched Protocol Answer: {best_row['answer']}\n"
            f"Intent: {best_row['intent_tag']}\n"
            f"Safety Flag: {best_row['safety_flag']}\n"
            f"Match Confidence: {confidence:.2%}"
        )

    history_lines = []
    for turn in history[-6:]:
        history_lines.append(f"Nurse: {turn['user']}")
        history_lines.append(f"Assistant: {turn['assistant']}")
    history_block = "\n".join(history_lines) if history_lines else "No prior turns."

    system_prompt = (
        "You are a clinical eMAR assistant for nursing workflows. "
        "Be conversational, concise, and clear. "
        "Use ONLY the provided protocol context for clinical facts. "
        "If confidence is low or context is missing, ask a clarifying question and advise escalation if risk exists. "
        "Never invent medical instructions."
    )

    user_prompt = (
        f"Conversation History:\n{history_block}\n\n"
        f"Protocol Context:\n{context_block}\n\n"
        f"Current Nurse Question:\n{user_text}\n\n"
        "Return a natural conversational reply. "
        "If protocol context is strong, provide the answer in plain nursing language."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def generate_conversational_response(user_text, best_row, confidence, history):
    client = get_openai_client()
    if client is not None:
        try:
            completion = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=build_genai_messages(user_text, best_row, confidence, history),
                temperature=0.2,
            )
            content = completion.choices[0].message.content.strip()
            if content:
                return content
        except Exception:
            pass

    if best_row is None:
        return (
            "I want to make sure I give the correct protocol. "
            "Please rephrase with key terms like medication name, alert name, or eMAR step."
        )

    if confidence >= 0.85:
        return f"Sure — {best_row['answer']}"
    if confidence >= 0.65:
        return f"Based on the closest protocol match: {best_row['answer']}"
    return (
        "I found a possible match, but confidence is limited. "
        f"Closest guidance is: {best_row['answer']}"
    )


def normalize_for_matching(text):
    compact = re.sub(r"\s+", " ", str(text or "").strip().lower())
    return re.sub(r"[^a-z0-9\s]", "", compact)


def is_definition_query(text):
    normalized = normalize_for_matching(text)
    return bool(
        re.search(
            r"\b(mean|means|meaning|what does|what is|stands for)\b",
            normalized,
        )
    )


def simplify_question_text(text):
    normalized = normalize_for_matching(text)
    normalized = re.sub(r"\blog\s*out\b", "logout", normalized)
    normalized = re.sub(r"\blog\s*off\b", "logout", normalized)
    normalized = re.sub(r"\bsign\s*out\b", "logout", normalized)
    normalized = re.sub(r"\bsign\s*off\b", "logout", normalized)
    normalized = re.sub(r"\be\s*mar\b", "emar", normalized)
    normalized = re.sub(r"\bwhat\s+does\s+", "", normalized)
    normalized = re.sub(r"\bwhat\s+is\s+the\s+meaning\s+if\s+", "", normalized)
    normalized = re.sub(r"\bwhat\s+is\s+the\s+meaning\s+of\s+", "", normalized)
    normalized = re.sub(r"\bwhat\s+is\s+", "", normalized)
    normalized = re.sub(r"\b(meaning|mean|means)\b", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def build_match_profile(text, intent_tag=""):
    raw_definition_query = is_definition_query(text)
    normalized_text = simplify_question_text(text)
    normalized_intent = normalize_for_matching(intent_tag)
    text_keywords = extract_keywords(normalized_text)
    intent_tokens = {token for token in normalized_intent.split() if len(token) > 1}
    combined_tokens = text_keywords | intent_tokens
    return {
        "normalized_text": normalized_text,
        "normalized_intent": normalized_intent,
        "tokens": combined_tokens,
        "token_count": len(text_keywords),
        "is_definition_query": raw_definition_query,
        "char_trigrams": Counter(
            normalized_text[index : index + 3]
            for index in range(max(0, len(normalized_text) - 2))
        ),
    }


def cosine_counter_similarity(counter_one, counter_two):
    if not counter_one or not counter_two:
        return 0.0
    shared_keys = set(counter_one.keys()) & set(counter_two.keys())
    numerator = sum(counter_one[key] * counter_two[key] for key in shared_keys)
    left = sum(value * value for value in counter_one.values()) ** 0.5
    right = sum(value * value for value in counter_two.values()) ** 0.5
    if left == 0 or right == 0:
        return 0.0
    return numerator / (left * right)


def blend_match_score(user_profile, row, user_doc=None):
    row_profile = row["match_profile"]
    row_keywords = row["keywords"]
    user_keywords = user_profile["tokens"]

    overlap = len(user_keywords & row_keywords)
    union_size = len(user_keywords | row_keywords)

    precision = overlap / max(1, len(row_keywords))
    recall = overlap / max(1, len(user_keywords))
    jaccard = overlap / max(1, union_size)
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    token_score = max(jaccard, f1_score)

    normalized_user = user_profile["normalized_text"]
    normalized_question = row_profile["normalized_text"]
    lexical_score = SequenceMatcher(None, normalized_user, normalized_question).ratio()
    char_ngram_score = cosine_counter_similarity(user_profile["char_trigrams"], row_profile["char_trigrams"])

    intent_score = SequenceMatcher(None, normalized_user, row_profile["normalized_intent"]).ratio()

    containment_bonus = 0.0
    if user_keywords and user_keywords.issubset(row_keywords):
        containment_bonus = 0.08

    definition_bonus = 0.0
    if user_profile["is_definition_query"]:
        if row_profile["is_definition_query"]:
            definition_bonus = 0.10
        else:
            definition_bonus = -0.06

    semantic_score = 0.0
    if user_doc is not None and nlp is not None and nlp.vocab.vectors_length > 0:
        semantic_score = user_doc.similarity(nlp(row["question"].lower()))
        if overlap == 0:
            semantic_score *= 0.55
        elif overlap == 1:
            semantic_score *= 0.80

    if semantic_score > 0:
        blended_score = (
            (0.38 * token_score)
            + (0.22 * lexical_score)
            + (0.20 * char_ngram_score)
            + (0.08 * intent_score)
            + (0.12 * semantic_score)
        )
    else:
        blended_score = (0.38 * token_score) + (0.26 * lexical_score) + (0.28 * char_ngram_score) + (0.08 * intent_score)

    return max(0.0, min(1.0, blended_score + containment_bonus + definition_bonus))


def calibrate_confidence(top_score, second_score):
    margin = max(0.0, top_score - second_score)
    if top_score < 0.45:
        return top_score * 0.60

    calibrated = (0.88 * top_score) + (0.12 * margin)

    if margin >= 0.20 and top_score >= 0.55:
        calibrated = min(0.99, calibrated + 0.05)
    elif margin < 0.05:
        calibrated = max(0.0, calibrated - 0.10)

    return max(0.0, min(0.99, calibrated))


def detect_conversation_intent(user_text):
    normalized = normalize_for_matching(user_text)
    if not normalized:
        return "empty"

    if re.search(r"\b(hi|hello|hey|good morning|good evening)\b", normalized):
        return "greeting"
    if re.search(r"\b(thanks|thank you|thx|appreciate)\b", normalized):
        return "thanks"
    if re.search(r"\b(help|how to use|what can you do)\b", normalized):
        return "help"
    if re.search(r"\b(bye|goodbye|see you)\b", normalized):
        return "bye"
    return "protocol"


def build_conversational_answer(match_row, confidence):
    core = generate_response(match_row)
    if confidence >= 0.90:
        return f"Here is the protocol: {core}"
    if confidence >= 0.75:
        return f"Based on the best match in the knowledge base: {core}"
    return f"This seems to match your query: {core}"


def rank_matches(cursor, user_text, kb_data):
    normalized_query = simplify_question_text(user_text)
    exact_row = next(
        (row for row in kb_data if row["match_profile"]["normalized_text"] == normalized_query),
        None,
    )

    if exact_row is not None:
        return {
            "best_row": exact_row,
            "confidence": 0.99,
            "second_score": 0.0,
            "min_confidence": 0.55,
            "top_rows": [(0.99, exact_row)],
        }

    user_profile = build_match_profile(user_text)
    user_doc = None
    if nlp is not None and nlp.vocab.vectors_length > 0:
        user_doc = nlp(user_text.lower())

    score_rows = []
    for row in kb_data:
        score = blend_match_score(user_profile, row, user_doc=user_doc)
        score_rows.append((score, row))

    if not score_rows:
        return {
            "best_row": None,
            "confidence": 0.0,
            "second_score": 0.0,
            "min_confidence": 0.55,
            "top_rows": [],
        }

    score_rows.sort(key=lambda item: item[0], reverse=True)
    max_score, best_row = score_rows[0]
    second_score = score_rows[1][0] if len(score_rows) > 1 else 0.0
    calibrated_confidence = calibrate_confidence(max_score, second_score)

    min_confidence = 0.62
    if user_profile["token_count"] <= 4:
        min_confidence = 0.55

    return {
        "best_row": best_row,
        "confidence": calibrated_confidence,
        "second_score": second_score,
        "min_confidence": min_confidence,
        "top_rows": score_rows[:3],
    }


def log_chat_interaction(user_query, bot_response, faq_knowledge_base_id=None):
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)
    session_id = get_or_create_session_id(cursor)
    log_query = """INSERT INTO chat_log
                   (user_query, bot_response, user_session_id, faq_knowledge_base_id)
                   VALUES (%s, %s, %s, %s)"""
    cursor.execute(log_query, (user_query, bot_response, session_id, faq_knowledge_base_id))
    conn.commit()
    conn.close()

def find_best_match(user_text):
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT id, question, answer, safety_flag, intent_tag FROM faq_knowledge_base")
    kb_rows = cursor.fetchall()

    kb_data = []
    for row in kb_rows:
        row["keywords"] = build_keywords(row["intent_tag"], row["question"])
        row["match_profile"] = build_match_profile(row["question"], row["intent_tag"])
        kb_data.append(row)

    rank_result = rank_matches(cursor, user_text, kb_data)
    best_row = rank_result["best_row"]
    calibrated_confidence = rank_result["confidence"]
    min_confidence = rank_result["min_confidence"]

    if best_row is None or calibrated_confidence < min_confidence:
        conn.close()
        return None, calibrated_confidence

    conn.close()
    return best_row, calibrated_confidence


def extract_keywords(text):
    if nlp is not None:
        doc = nlp(text.lower())
        return {token.lemma_ for token in doc if token.is_alpha and not token.is_stop}
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    return {token for token in tokens if len(token) > 2}


def build_keywords(intent_tag, question):
    intent_tokens = re.split(r"[^a-zA-Z]+", str(intent_tag).lower())
    intent_keywords = {token for token in intent_tokens if token}
    return intent_keywords | extract_keywords(question)


def get_or_create_session_id(cursor):
    cursor.execute("SELECT id FROM user_session ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    if row:
        return row["id"] if isinstance(row, dict) else row[0]

    session_key = str(uuid.uuid4())
    cursor.execute(
        "INSERT INTO user_session (user_id, session_key) VALUES (%s, %s)",
        ("system", session_key),
    )
    return cursor.lastrowid


def generate_response(row):
    intent_label = row["intent_tag"].replace("_", " ").title()
    if row["safety_flag"] == 1:
        return f"Critical protocol ({intent_label}): {row['answer']}"
    return f"Guidance ({intent_label}): {row['answer']}"

def run_chat_interface():
    print("Nurse Assistant Bot is Active.")
    conversation_history = []
    while True:
        input_text = input("\nNurse: ")
        if input_text.lower() in ['exit', 'quit']:
            break

        intent = detect_conversation_intent(input_text)
        if intent == "empty":
            assistant_reply = "Please type your medication or eMAR question, and I will help."
            print(f"Assistant: {assistant_reply}")
            conversation_history.append({"user": input_text, "assistant": assistant_reply})
            continue
        if intent == "greeting":
            assistant_reply = "Hello. I can help with eMAR protocols, medication safety, alerts, and documentation steps."
            print(f"Assistant: {assistant_reply}")
            conversation_history.append({"user": input_text, "assistant": assistant_reply})
            continue
        if intent == "thanks":
            assistant_reply = "You are welcome. Ask another medication or eMAR question any time."
            print(f"Assistant: {assistant_reply}")
            conversation_history.append({"user": input_text, "assistant": assistant_reply})
            continue
        if intent == "help":
            assistant_reply = "Ask me any eMAR or medication protocol question, for example: 'What does Pending PRN Outcomes alert mean?'"
            print(f"Assistant: {assistant_reply}")
            conversation_history.append({"user": input_text, "assistant": assistant_reply})
            continue
        if intent == "bye":
            assistant_reply = "Goodbye. Stay safe on shift."
            print(f"Assistant: {assistant_reply}")
            conversation_history.append({"user": input_text, "assistant": assistant_reply})
            break

        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, question, answer, safety_flag, intent_tag FROM faq_knowledge_base")
        kb_rows = cursor.fetchall()
        kb_data = []
        for row in kb_rows:
            row["keywords"] = build_keywords(row["intent_tag"], row["question"])
            row["match_profile"] = build_match_profile(row["question"], row["intent_tag"])
            kb_data.append(row)

        rank_result = rank_matches(cursor, input_text, kb_data)
        best_row = rank_result["best_row"]
        confidence = rank_result["confidence"]
        min_confidence = rank_result["min_confidence"]
        second_score = rank_result["second_score"]
        top_rows = rank_result["top_rows"]

        if best_row and confidence >= min_confidence:
            assistant_reply = generate_conversational_response(input_text, best_row, confidence, conversation_history)
            print(f"Assistant: {assistant_reply}")
            print(f"(Confidence: {confidence:.2%})")
            conn.close()
            log_chat_interaction(input_text, assistant_reply, best_row["id"])
            conversation_history.append({"user": input_text, "assistant": assistant_reply})
            continue

        ambiguous = best_row is not None and (confidence >= 0.50) and ((confidence - second_score) <= 0.08)
        if ambiguous and len(top_rows) >= 2:
            option_one = top_rows[0][1]["question"]
            option_two = top_rows[1][1]["question"]
            assistant_reply = (
                "I found two close protocols. Please confirm which one you mean:\n"
                f"1) {option_one}\n"
                f"2) {option_two}"
            )
            print(f"Assistant: {assistant_reply}")
            conn.close()
            log_chat_interaction(input_text, assistant_reply, None)
            conversation_history.append({"user": input_text, "assistant": assistant_reply})
            continue

        assistant_reply = generate_conversational_response(input_text, best_row, confidence, conversation_history)
        print(f"Assistant: {assistant_reply}")
        conn.close()
        log_chat_interaction(input_text, assistant_reply, best_row["id"] if best_row else None)
        conversation_history.append({"user": input_text, "assistant": assistant_reply})


if __name__ == "__main__":
    run_chat_interface()
