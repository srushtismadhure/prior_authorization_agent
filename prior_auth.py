# prior_auth_pipeline.py
import os
import pandas as pd
import spacy
from rapidfuzz import process, fuzz
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
import os



# -------------------------------
# Setup
# -------------------------------

# Load environment variables from .env file
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Missing OPENAI_API_KEY in .env file.")


# Initialize OpenAI client
client = OpenAI(api_key=api_key)
nlp = spacy.load("en_core_web_sm")

# -------------------------------
# Load Data
# -------------------------------

notes = pd.read_csv("clinical.csv")
carrier = pd.read_csv("prior_auth.csv")
print(f"Loaded files: notes={notes.shape}, carrier={carrier.shape}")



# -------------------------------
# Clean and Prepare Carrier Data
# -------------------------------
carrier.columns = (
    carrier.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

carrier = carrier[["code", "description_of_service", "service_category", "approval_rate"]]

# Add missing common services
additional_services = [
    {"code": "78815", "description_of_service": "pet scan whole body", "service_category": "Imaging", "approval_rate": "85%"},
    {"code": "78816", "description_of_service": "pet scan skull base to mid thigh", "service_category": "Imaging", "approval_rate": "85%"},
    {"code": "95810", "description_of_service": "sleep study", "service_category": "Sleep Medicine", "approval_rate": "90%"},
    {"code": "95811", "description_of_service": "polysomnography", "service_category": "Sleep Medicine", "approval_rate": "90%"},
]
carrier = pd.concat([carrier, pd.DataFrame(additional_services)], ignore_index=True)
carrier["description_of_service"] = carrier["description_of_service"].astype(str).str.lower()

print("Carrier data cleaned and enriched.")
print(carrier.head(5))


# -------------------------------
# Matching Function
# -------------------------------
def find_related_services(note: str):
    """Fuzzy match clinical note text to carrier service descriptions."""
    note = note.lower()
    match = process.extractOne(note, carrier["description_of_service"], scorer=fuzz.partial_ratio)

    if match and match[1] > 80:
        text, score, idx = match
        row = carrier.iloc[idx]
        return {
            "Matched_Description": row["description_of_service"],
            "Code": row["code"],
            "Category": row["service_category"],
            "Approval_Rate": row["approval_rate"],
            "Confidence_Score": score
        }
    return None

# -------------------------------
# Apply Matching
# -------------------------------
print(" Matching services to clinical notes...")
notes["Matched_Service"] = notes["Clinical_Notes"].astype(str).apply(find_related_services)
matched_info = notes["Matched_Service"].apply(pd.Series)
notes = pd.concat([notes, matched_info], axis=1)
notes["Needs_Prior_Auth"] = notes["Matched_Description"].notnull()


# -------------------------------
# Save Matched Results
# -------------------------------
output_csv = "matched_notes_output.csv"
notes.to_csv(output_csv, index=False)
print(f"Matched results saved to {output_csv}")

# -------------------------------
# Summarization Function
# -------------------------------
def summarize(row):
    """Generate a short summary using OpenAI GPT."""
    if row["Needs_Prior_Auth"]:
        prompt = f"""
        Patient note: {row['Clinical_Notes']}
        Service: {row['Matched_Description']}
        Code: {row['Code']}
        Category: {row['Category']}
        Approval Rate: {row['Approval_Rate']}

        Summarize the patient's prior authorization requirement in 2 sentences.
        """
    else:
        prompt = f"Patient note: {row['Clinical_Notes']}\nSummarize briefly. No prior authorization required."

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# -------------------------------
# Summarize Sample or All
# -------------------------------
print("Generating summaries for matched notes...")
tqdm.pandas()  # enables progress bar for apply

notes["Summary"] = notes.progress_apply(summarize, axis=1)

# -------------------------------
# Save as CSV and TXT
# -------------------------------
notes.to_csv("final_output.csv", index=False)

with open("final_output.txt", "w", encoding="utf-8") as f:
    for _, row in notes.iterrows():
        f.write(f"Clinical Note: {row['Clinical_Notes']}\n")
        f.write(f"Matched Description: {row['Matched_Description']}\n")
        f.write(f"Needs Prior Auth: {row['Needs_Prior_Auth']}\n")
        f.write(f"Summary: {row['Summary']}\n")
        f.write("-" * 80 + "\n")

print("Final results saved to final_output.csv and final_output.txt")









