import redis
import json
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

class RedisFeatureStore:
    def __init__(self, host="localhost", port=6379, db=0):
        try:
            self.client = redis.StrictRedis(
                host=host,
                port=port,
                db=db,
                decode_responses=True
            )
            # Test connection
            self.client.ping()
            logger.info(f"Redis connection established → {host}:{port}")
        except Exception as e:
            logger.error(f"Failed to establish Redis connection: {e}")
            raise CustomException("Failed to establish Redis connection", e)

    def store_features(self, entity_id, features):
        """It writes a single record to Redis."""
        key = f"entity:{entity_id}:features"
        self.client.set(key, json.dumps(features))

    def get_features(self, entity_id):
        """It reads a single record from Redis."""
        key = f"entity:{entity_id}:features"
        features = self.client.get(key)
        return json.loads(features) if features else None

    def store_batch_features(self, batch_data):
        """It writes multiple records to Redis."""
        for entity_id, features in batch_data.items():
            self.store_features(entity_id, features)
        logger.info(f"{len(batch_data)} records written to Redis.")

    def get_batch_features(self, entity_ids):
        """It reads multiple records from Redis."""
        return {eid: self.get_features(eid) for eid in entity_ids}

    def get_all_entity_ids(self):
        """It returns all entity IDs in Redis."""
        keys = self.client.keys('entity:*:features')
        return [key.split(':')[1] for key in keys]

    def flush(self):
        """It clears all Redis data (use with caution)."""
        self.client.flushdb()
        logger.info("Redis database cleared.")