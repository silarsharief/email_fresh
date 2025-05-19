from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import os
from send_sample_email import send_email  # Import your existing email sending function

app = FastAPI(title="Email Bot API")

@app.get("/")
async def root():
    return {"message": "Email Bot API is running"}

@app.post("/send-email")
async def trigger_email():
    try:
        # Call your existing email sending function
        send_email()
        return JSONResponse(
            status_code=200,
            content={"message": "Email sent successfully"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Get the port from environment variable or default to 8080
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port) 