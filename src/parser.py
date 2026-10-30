import email
from email import policy
from bs4 import BeautifulSoup
import re


def extract_email(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        msg = email.message_from_file(f, policy=policy.default)

    subject = msg['subject']
    sender = msg['from']

    body = ""

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                body += part.get_content()
            elif content_type == "text/html":
                html = part.get_content()
                soup = BeautifulSoup(html, "html.parser")
                body += soup.get_text()
    else:
        body = msg.get_content()

    urls = re.findall(r'https?://[^\s]+', body)

    return {
        "subject": subject,
        "sender": sender,
        "body": body,
        "urls": urls
    }