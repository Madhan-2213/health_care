import re

import mysql.connector
import pandas as pd

try:
    import spacy
except ImportError:
    spacy = None

db_config = {
    "host": "localhost",
    "user": "root",
    "password": "mayura_ziani",
    "database": "health_care_chatbot",
}


def load_nlp_pipeline():
    if spacy is None:
        return None

    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        return nlp


nlp = load_nlp_pipeline()


def normalize_text(value):
    if pd.isna(value):
        return ""
    text = str(value)
    return re.sub(r"\s+", " ", text).strip()


def to_actionable_instruction(answer_text):
    text = normalize_text(answer_text)
    if not text:
        return ""

    if nlp is not None:
        doc = nlp(text)
        first_sentence = next(doc.sents, None)
        candidate = first_sentence.text.strip() if first_sentence is not None else text
    else:
        sentence_parts = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)
        candidate = sentence_parts[0].strip() if sentence_parts else text

    candidate = re.sub(
        r"^(it\s+(is|helps|improves|shows|means|displays)\s+)",
        "",
        candidate,
        flags=re.IGNORECASE,
    )

    if not re.match(
        r"^(check|confirm|review|record|select|administer|delay|inform|document|escalate|monitor|log|verify|follow|continue|ensure|mark|use)\b",
        candidate,
        flags=re.IGNORECASE,
    ):
        candidate = f"Follow this guidance: {candidate}"

    return candidate.rstrip(" .") + "."


def prepare_table_for_idempotent_import(cursor):
    cursor.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = 'faq_knowledge_base'
          AND COLUMN_NAME = 'source_hash'
        """
    )
    has_source_hash = cursor.fetchone()[0] > 0
    if not has_source_hash:
        cursor.execute("ALTER TABLE faq_knowledge_base ADD COLUMN source_hash CHAR(64) NULL")

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.STATISTICS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = 'faq_knowledge_base'
          AND INDEX_NAME = 'uq_faq_source_hash'
        """
    )
    has_unique_index = cursor.fetchone()[0] > 0
    if has_unique_index:
        cursor.execute("DROP INDEX uq_faq_source_hash ON faq_knowledge_base")

    cursor.execute(
        """
        DELETE f1
        FROM faq_knowledge_base f1
        JOIN faq_knowledge_base f2
          ON TRIM(f1.intent_tag) = TRIM(f2.intent_tag)
         AND TRIM(f1.question) = TRIM(f2.question)
         AND f1.id > f2.id
        """
    )

    cursor.execute(
        """
        UPDATE faq_knowledge_base
        SET source_hash = SHA2(CONCAT_WS('||', TRIM(intent_tag), TRIM(question)), 256)
        """
    )
    cursor.execute("CREATE UNIQUE INDEX uq_faq_source_hash ON faq_knowledge_base (source_hash)")


def upload_perfect_data(file_path):
    conn = None
    cursor = None
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        prepare_table_for_idempotent_import(cursor)

        df = pd.read_csv(file_path)
        print(f"Detected Headers: {list(df.columns)}")

        required_columns = {"Question", "Answer", "Intent_Tag", "Safety_Flag"}
        missing_columns = required_columns.difference(df.columns)
        if missing_columns:
            raise ValueError(f"CSV is missing required columns: {sorted(missing_columns)}")

        df = df.dropna(subset=["Question", "Answer", "Intent_Tag", "Safety_Flag"]).copy()
        df["Question"] = df["Question"].map(normalize_text)
        df["Answer"] = df["Answer"].map(to_actionable_instruction)
        df["Intent_Tag"] = df["Intent_Tag"].map(normalize_text)
        df["Safety_Flag"] = pd.to_numeric(df["Safety_Flag"], errors="coerce").fillna(0).astype(int).clip(0, 1)

        df = df.drop_duplicates(subset=["Intent_Tag", "Question", "Answer", "Safety_Flag"], keep="first")
        print(f"Processing {len(df)} unique medical records...")

        sql = """
            INSERT INTO faq_knowledge_base (intent_tag, question, answer, safety_flag, source_hash)
            VALUES (%s, %s, %s, %s, SHA2(CONCAT_WS('||', TRIM(%s), TRIM(%s)), 256))
            ON DUPLICATE KEY UPDATE
                intent_tag = VALUES(intent_tag),
                question = VALUES(question),
                answer = VALUES(answer),
                safety_flag = VALUES(safety_flag)
        """

        records = [
            (
                row["Intent_Tag"],
                row["Question"],
                row["Answer"],
                int(row["Safety_Flag"]),
                row["Intent_Tag"],
                row["Question"],
            )
            for _, row in df.iterrows()
        ]

        cursor.executemany(sql, records)
        conn.commit()
        print("✅ SUCCESS: FAQ import is now idempotent (no duplicate rows on re-run).")

    except Exception as e:
        print(f"❌ ERROR: {e}")
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None and conn.is_connected():
            conn.close()


upload_perfect_data(
    "C:\\Users\\admin\\Downloads\\HealthCare Chatbot\\medication_bot\\FAQ file\\Master_Categorized_FAQ.csv"
)
