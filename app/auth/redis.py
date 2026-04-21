# app/auth/redis.py
from app.core.config import get_settings

settings = get_settings()
_fallback_blacklist = set()

async def get_redis():
    if hasattr(get_redis, "redis"):
        return get_redis.redis

    try:
        from redis import asyncio as redis_asyncio
        get_redis.redis = await redis_asyncio.from_url(
            settings.REDIS_URL or "redis://localhost"
        )
        return get_redis.redis
    except Exception:
        return None

async def add_to_blacklist(jti: str, exp: int):
    """Add a token's JTI to the blacklist"""
    redis = await get_redis()
    if redis is None:
        _fallback_blacklist.add(jti)
        return
    await redis.set(f"blacklist:{jti}", "1", ex=exp)

async def is_blacklisted(jti: str) -> bool:
    """Check if a token's JTI is blacklisted"""
    redis = await get_redis()
    if redis is None:
        return jti in _fallback_blacklist
    return bool(await redis.exists(f"blacklist:{jti}"))