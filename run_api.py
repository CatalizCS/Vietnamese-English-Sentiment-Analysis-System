
import uvicorn
from src.config import Config

def main():
    config = Config()
    uvicorn.run(
        "src.api.app:app",
        host=config.API_CONFIG['HOST'],
        port=config.API_CONFIG['PORT'],
        workers=config.API_CONFIG['WORKERS'],
        timeout=config.API_CONFIG['TIMEOUT'],
        reload=config.API_CONFIG['RELOAD']
    )

if __name__ == "__main__":
    main()