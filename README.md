# Email Bot

An automated email bot that sends financial market updates using FastAPI and Google Cloud Run.

## Setup Instructions

### 1. Google Cloud Setup

1. Create a new project in Google Cloud Console
2. Enable required APIs:
   - Cloud Run API
   - Cloud Build API
   - Container Registry API
   - Secret Manager API (optional, for storing credentials)

### 2. Service Account Setup

1. Go to IAM & Admin > Service Accounts
2. Create a new service account with the following roles:
   - Cloud Run Admin
   - Storage Admin
   - Service Account User
3. Create a new key (JSON format) for this service account
4. Download the key file

### 3. GitHub Repository Setup

1. Create a new repository on GitHub
2. Add the following secrets to your repository (Settings > Secrets and variables > Actions):
   - `GCP_PROJECT_ID`: Your Google Cloud project ID
   - `GCP_SA_KEY`: The entire contents of your service account key JSON file

### 4. Deploy

1. Push your code to the main branch:
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

2. The GitHub Action will automatically:
   - Build your Docker container
   - Push it to Google Container Registry
   - Deploy it to Cloud Run

### 5. Schedule Email Sending

After deployment, set up Cloud Scheduler:

1. Go to Cloud Scheduler in Google Cloud Console
2. Create a new job:
   - Frequency: `0 9 * * *` (runs at 9 AM daily)
   - Target type: HTTP
   - URL: Your Cloud Run service URL + `/send-email`
   - HTTP method: POST

## Local Development

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python email/app.py
```

## Environment Variables

Create a `.env` file with the following variables:
```
EMAIL_USER=your-email@gmail.com
EMAIL_PASSWORD=your-app-specific-password
```

Note: For Gmail, you'll need to use an App Password. Enable 2-factor authentication and generate an App Password from your Google Account settings.

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