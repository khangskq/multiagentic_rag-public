from typing import Any, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path

class NewsCache:
    def __init__(self, cache_dir: str = "cache", expiry_hours: int = 1):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.expiry = timedelta(hours=expiry_hours)
    
    def get(self, key: str) -> Optional[Any]:
        cache_file = self.cache_dir / f"{key}.json"
        if not cache_file.exists():
            return None
            
        data = json.loads(cache_file.read_text())
        if datetime.fromisoformat(data['timestamp']) + self.expiry < datetime.now():
            cache_file.unlink()
            return None
            
        return data['content']
    
    def set(self, key: str, value: Any) -> None:
        cache_file = self.cache_dir / f"{key}.json"
        data = {
            'timestamp': datetime.now().isoformat(),
            'content': value
        }
        cache_file.write_text(json.dumps(data))