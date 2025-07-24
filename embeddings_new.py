import os
import re
import json
import pickle
import faiss
from sentence_transformers import SentenceTransformer

# Configurations
BASE_DIR = "./EDA"
PKL_OUTPUT = "final_output.pkl"
INDEX_OUTPUT = "final_index.index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

model = SentenceTransformer(EMBEDDING_MODEL)
all_texts = []
metadata_list = []

def embed_text(text):
    return model.encode([text])[0]

def safe_read_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(filepath, "r", encoding="latin-1") as f:
            return f.read()

# 1. Process EDA/eda_insights_report.txt
eda_insight_path = os.path.join(BASE_DIR, "eda_insights_report.txt")
if os.path.isfile(eda_insight_path):
    content = safe_read_file(eda_insight_path)
    all_texts.append(content)
    metadata_list.append((content, {'file_path': eda_insight_path, 'document_type': 'eda_insight'}))

# 2. Process EDA/statistics (.txt and .json files)
statistics_dir = os.path.join(BASE_DIR, "statistics")
if os.path.isdir(statistics_dir):
    for fname in os.listdir(statistics_dir):
        fpath = os.path.join(statistics_dir, fname)
        if fname.endswith(".txt"):
            content = safe_read_file(fpath)
            all_texts.append(content)
            metadata_list.append((content, {'file_path': fpath, 'document_type': 'statistics_txt'}))
        elif fname.endswith(".json"):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    json_data = json.load(f)
                # Convert JSON to a readable string format
                content = json.dumps(json_data, indent=2)
                all_texts.append(content)
                metadata_list.append((content, {'file_path': fpath, 'document_type': 'statistics_json'}))
            except Exception as e:
                print(f"Error reading JSON file {fpath}: {e}")

# 3. Process EDA/summaries
summaries_dir = os.path.join(BASE_DIR, "summaries")
if os.path.isdir(summaries_dir):
    for fname in os.listdir(summaries_dir):
        fpath = os.path.join(summaries_dir, fname)
        content = safe_read_file(fpath)
        all_texts.append(content)
        metadata_list.append((content, {'file_path': fpath, 'document_type': 'eda_summary'}))

# 4. Process LLM_summaries
llm_dir = "./LLM_summaries"
if os.path.isdir(llm_dir):
    for fname in os.listdir(llm_dir):
        fpath = os.path.join(llm_dir, fname)
        content = safe_read_file(fpath)
        all_texts.append(content)
        metadata_list.append((content, {'file_path': fpath, 'document_type': 'llm_summary'}))

# 5. Process schema_details
schema_dir = "./schema_details"
schema_pattern = re.compile(
    r"Table: (?P<table_name>.*?)\nColumns:\n(?P<columns>.*?)(Primary Keys|Foreign Keys)",
    re.DOTALL
)
pk_pattern = re.compile(r"Primary Keys:\n(.*?)(Foreign Keys|$)", re.DOTALL)
fk_pattern = re.compile(r"Foreign Keys:\n(.*?)(\n[-=]{10,}|$)", re.DOTALL)

def parse_columns(column_text):
    cols = []
    types = {}
    lines = [line.strip() for line in column_text.strip().split("\n") if line.strip()]
    for line in lines:
        if line.startswith("- "):
            col = line.replace("- ", "").strip()
            colname, coltype = col.split(" (")
            coltype = coltype.strip(")")
            cols.append(colname.strip())
            types[colname.strip()] = coltype.strip()
    return cols, types

def parse_relationships(fk_text):
    rels = []
    lines = [line.strip() for line in fk_text.strip().split("\n") if line.strip()]
    for line in lines:
        if "Constraint" in line:
            rels.append(line)
    return rels

if os.path.isdir(schema_dir):
    for fname in os.listdir(schema_dir):
        fpath = os.path.join(schema_dir, fname)
        content = safe_read_file(fpath)

        tables = content.split("--------------------------------------------------")
        for table_block in tables:
            if "Table:" in table_block:
                table_match = schema_pattern.search(table_block)
                if table_match:
                    table_name = table_match.group('table_name').strip()
                    col_text = table_match.group('columns')

                    columns, column_types = parse_columns(col_text)

                    pk_match = pk_pattern.search(table_block)
                    pk_list = [pk.strip("- ").strip() for pk in pk_match.group(1).strip().split("\n") if pk.strip()] if pk_match else []

                    fk_match = fk_pattern.search(table_block)
                    relationships = parse_relationships(fk_match.group(1)) if fk_match else []

                    final_text = f"{table_name}\nColumns:\n"
                    for col in columns:
                        final_text += f"  - {col} ({column_types[col]})\n"
                    final_text += "Primary Keys:\n  - " + "\n  - ".join(pk_list) if pk_list else "Primary Keys: None"
                    final_text += "\nForeign Keys: " + (", ".join(relationships) if relationships else "None") + "\n\n" + "-"*50 + "\n"

                    metadata = {
                        'table_name': table_name,
                        'document_type': 'schema',
                        'column_names': columns,
                        'column_types': column_types,
                        'relationships': relationships,
                        'file_path': fpath
                    }

                    all_texts.append(final_text)
                    metadata_list.append((final_text, metadata))

# Generate embeddings
print("Generating embeddings...")
embeddings = model.encode(all_texts, show_progress_bar=True)

# Save PKL
with open(PKL_OUTPUT, "wb") as pkl_file:
    pickle.dump(metadata_list, pkl_file)
print(f"Metadata pickle saved: {PKL_OUTPUT}")

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
faiss.write_index(index, INDEX_OUTPUT)
print(f"FAISS index saved: {INDEX_OUTPUT}")

print(f"Total documents processed: {len(all_texts)}")
print(f"Embedding dimension: {dimension}")
print(f"Embeddings shape: {embeddings.shape}")
