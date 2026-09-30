"""Rebuild the configured collection from PDFs using the existing ingest pipeline."""

from src.config import settings
from src.indexing import ingest
from src.store import close_client, get_client
from qdrant_client.local.qdrant_local import QdrantLocal


def main() -> None:
    try:
        client = get_client()
        # Local Qdrant can leave SQLite files locked during deletion on Windows.
        # Close the target's handle so the existing recreate flow can remove it.
        if isinstance(client._client, QdrantLocal):
            collection = client._client.collections.get(settings.qdrant_collection)
            if collection is not None:
                collection.close()
        count = ingest(recreate=True)
        print(f"Number of chunks ingested: {count}")
        point_count = get_client().count(
            collection_name=settings.qdrant_collection, exact=True
        ).count
        print(f"Number of points in collection: {point_count}")
        assert point_count == count, "Collection point count differs from ingested chunk count."
    finally:
        close_client()


if __name__ == "__main__":
    main()
