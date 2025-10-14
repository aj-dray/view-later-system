import asyncio
import sys
from pathlib import Path
from pprint import pprint
import json

# Add the server directory to the path so we can import from app
server_dir = Path(__file__).parent.parent
sys.path.insert(0, str(server_dir))

from app.services import extract_data


async def exp_extraction(url):
    """Run the extraction and dump the result for manual inspection."""
    result = await extract_data(url)
    from datetime import datetime

    metadata = {k: v for k, v in result.items() if not k.startswith("content_")}

    # Convert any datetime objects in metadata to ISO format strings
    for k, v in metadata.items():
        if isinstance(v, datetime):
            metadata[k] = v.isoformat()

    pprint(metadata)
    metadata_path = Path("backend/tests/_data/metadata.json")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    markdown = result["content_markdown"]
    print(markdown)
    if markdown is not None:
        markdown_path = Path("backend/tests/_data/markdown.md")
        markdown_path.write_text(markdown, encoding="utf-8")

    text = result["content_text"]
    print(text)
    text_path = Path("backend/tests/_data/text.txt")
    text_path.write_text(text, encoding="utf-8")



if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python3 script.py <url>")
        sys.exit(1)
    url = sys.argv[1]

    asyncio.run(exp_extraction(url))
