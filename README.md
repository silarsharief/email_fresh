# Email Market Digest Bot

An automated email bot that sends market digest reports with Nifty50 and Sensex data visualization.

## Features
- Automated market data collection
- HTML email generation with embedded graphs
- Secure email delivery using Gmail SMTP
- Customizable email templates

## Project Structure
```
email_bot/
├── email/
│   ├── send_sample_email.py    # Main email sending script
│   ├── render_email.py         # Email template rendering
│   ├── rendered_email.html     # Email template
│   ├── nifty50_graph.png       # Market visualization
│   ├── sensex_graph.png        # Market visualization
│   └── twlogo.png             # Company logo
├── requirements.txt            # Project dependencies
├── .env.example               # Environment variables template
└── README.md                  # Project documentation
```

## Setup
1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `.\venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Create a `.env` file with your email credentials:
   ```
   EMAIL_ADDRESS=your_email@gmail.com
   EMAIL_PASSWORD=your_app_password
   RECIPIENT_EMAILS=email1@example.com,email2@example.com
   ```

## Usage
Run the email sender:
```bash
python email/send_sample_email.py
```

## Deployment
This project is configured for deployment on Google Cloud Platform.

## Security Note
- Never commit your `.env` file or any sensitive credentials
- Use app passwords for Gmail authentication
- Keep your API keys secure 