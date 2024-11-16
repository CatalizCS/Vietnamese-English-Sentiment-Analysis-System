import uvicorn
from src.config import Config

def main():
    config = Config()
    # Only use supported uvicorn.run() parameters
    uvicorn.run(
        "src.api.app:app",
        host=config.API_CONFIG['HOST'],
        port=config.API_CONFIG['PORT'],
        workers=config.API_CONFIG['WORKERS'],
        reload=config.API_CONFIG['RELOAD'],
        log_level="info",
        limit_concurrency=100
    )

if __name__ == "__main__":
    main()