from dataclasses import dataclass
from pathlib import Path
import requests


@dataclass
class DataFetcher:
    url: str
    out_path: Path

    def fetch(self) -> Path:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

        r = requests.get(self.url, timeout=30)
        r.raise_for_status()

        self.out_path.write_bytes(r.content)

        # Basic sanity check: avoid saving an HTML error page as "csv"
        if self.out_path.stat().st_size < 500:
            raise ValueError(
                "Downloaded file is unexpectedly small. "
                "Check the URL or whether the file is public."
            )

        return self.out_path
