from fastapi import FastAPI

app = FastAPI()

# Initialize global variable
config = {}

@app.on_event("startup")
async def startup_event():
    # Load configurations or initialize resources here
    config["message"] = "Hello, FastAPI!"
    print("Application startup: Configurations loaded")

@app.get("/")
async def read_root():
    return {"message": config.get("message", "No message set")}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    print("Running the application...")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
