import os
import email
import pandas as pd
from email import policy

# -------------------------
# Dataset Paths
# -------------------------

DATASETS = {
    "easy_ham": 0,
    "easy_ham_2": 0,
    "spam": 1,
    "spam_2": 1
}

BASE_PATH = "data/raw/spamassassin"

emails = []

# -------------------------
# Read Emails
# -------------------------

for folder, label in DATASETS.items():

    folder_path = os.path.join(BASE_PATH, folder)

    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        continue

    for filename in os.listdir(folder_path):

        file_path = os.path.join(folder_path, filename)

        if os.path.isdir(file_path):
            continue

        try:

            with open(file_path, "r", encoding="latin1", errors="ignore") as f:

                msg = email.message_from_file(
                    f,
                    policy=policy.default
                )

                subject = msg["subject"]

                body = ""

                if msg.is_multipart():

                    for part in msg.walk():

                        if part.get_content_type() == "text/plain":

                            body += part.get_content()

                else:

                    body = msg.get_content()

                emails.append({
                    "subject": subject,
                    "body": body,
                    "label": label
                })

        except Exception as e:

            print("Skipped:", filename)

# -------------------------
# Save Dataset
# -------------------------

df = pd.DataFrame(emails)

output_path = "data/processed/emails.csv"

df.to_csv(output_path, index=False)

print(df.head())

print()

print("Total Emails:", len(df))

print()

print("Saved to:", output_path)