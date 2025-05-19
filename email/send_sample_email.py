import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
sender_email = os.getenv('EMAIL_ADDRESS')
sender_password = os.getenv('EMAIL_PASSWORD')
recipient_emails = os.getenv('RECIPIENT_EMAILS', '').split(',')

# Get current date in the format "Month DD, YYYY"
current_date = datetime.now().strftime("%B %d, %Y")
subject = f"Tw Market Digest on {current_date}"

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Read your rendered HTML
with open(os.path.join(script_dir, "rendered_email.html"), "r", encoding="utf-8") as f:
    html_content = f.read()

# Replace the image src in HTML to use CID for both the graph and the logo
html_content = html_content.replace('src="nifty50_graph.png"', 'src="cid:nifty50graph"')
html_content = html_content.replace('src="twlogo.png"', 'src="cid:twlogo"')

# --- SEND EMAIL ---
for recipient_email in recipient_emails:
    # Create the email for each recipient
    msg = MIMEMultipart("related")
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = recipient_email

    # Attach the HTML
    msg_alt = MIMEMultipart("alternative")
    msg.attach(msg_alt)
    msg_alt.attach(MIMEText(html_content, "html"))

    # Attach the image
    with open(os.path.join(script_dir, "nifty50_graph.png"), "rb") as img:
        mime_img = MIMEImage(img.read())
        mime_img.add_header('Content-ID', '<nifty50graph>')
        mime_img.add_header('Content-Disposition', 'inline', filename="nifty50_graph.png")
        msg.attach(mime_img)

    # Attach the logo image
    with open(os.path.join(script_dir, "twlogo.png"), "rb") as logo_img:
        mime_logo = MIMEImage(logo_img.read())
        mime_logo.add_header('Content-ID', '<twlogo>')
        mime_logo.add_header('Content-Disposition', 'inline', filename="twlogo.png")
        msg.attach(mime_logo)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())
    print(f"Email sent to {recipient_email}!")