import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime
import os
from dotenv import load_dotenv

def send_email():
    try:
        # Load environment variables
        load_dotenv()

        # --- CONFIGURATION ---
        sender_email = os.getenv('EMAIL_ADDRESS')
        sender_password = os.getenv('EMAIL_PASSWORD')
        recipient_emails = os.getenv('RECIPIENT_EMAILS', '').split(',')

        if not all([sender_email, sender_password, recipient_emails]):
            raise ValueError("Missing required environment variables: EMAIL_ADDRESS, EMAIL_PASSWORD, or RECIPIENT_EMAILS")

        # Get current date in the format "Month DD, YYYY"
        current_date = datetime.now().strftime("%B %d, %Y")
        subject = f"Tw Market Digest on {current_date}"

        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Read your rendered HTML
        html_path = os.path.join(script_dir, "rendered_email.html")
        if not os.path.exists(html_path):
            raise FileNotFoundError(f"Email template not found at {html_path}")

        with open(html_path, "r", encoding="utf-8") as f:
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
            nifty_path = os.path.join(script_dir, "nifty50_graph.png")
            if os.path.exists(nifty_path):
                with open(nifty_path, "rb") as img:
                    mime_img = MIMEImage(img.read())
                    mime_img.add_header('Content-ID', '<nifty50graph>')
                    mime_img.add_header('Content-Disposition', 'inline', filename="nifty50_graph.png")
                    msg.attach(mime_img)

            # Attach the logo image
            logo_path = os.path.join(script_dir, "twlogo.png")
            if os.path.exists(logo_path):
                with open(logo_path, "rb") as logo_img:
                    mime_logo = MIMEImage(logo_img.read())
                    mime_logo.add_header('Content-ID', '<twlogo>')
                    mime_logo.add_header('Content-Disposition', 'inline', filename="twlogo.png")
                    msg.attach(mime_logo)

            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, recipient_email, msg.as_string())
            print(f"Email sent to {recipient_email}!")
        
        return True
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        raise