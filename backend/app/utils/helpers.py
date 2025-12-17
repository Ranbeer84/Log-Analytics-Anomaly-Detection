"""
Common helper utilities
"""
import hashlib
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from functools import wraps
import asyncio


def generate_hash(data: Union[str, Dict, List]) -> str:
    """
    Generate SHA256 hash of data
    
    Args:
        data: Data to hash
        
    Returns:
        Hex digest of hash
    """
    if isinstance(data, (dict, list)):
        data = json.dumps(data, sort_keys=True)
    
    return hashlib.sha256(data.encode()).hexdigest()


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate string to maximum length
    
    Args:
        text: String to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncated
        
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split list into chunks
    
    Args:
        lst: List to split
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten nested dictionary
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for nested keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    
    return dict(items)


def get_time_bucket(timestamp: datetime, interval_seconds: int) -> datetime:
    """
    Round timestamp down to time bucket
    
    Args:
        timestamp: Timestamp to round
        interval_seconds: Bucket interval in seconds
        
    Returns:
        Rounded timestamp
    """
    unix_time = int(timestamp.timestamp())
    bucket_time = (unix_time // interval_seconds) * interval_seconds
    return datetime.fromtimestamp(bucket_time)


def parse_interval_to_seconds(interval: str) -> int:
    """
    Parse interval string to seconds
    
    Args:
        interval: Interval string (e.g., "5m", "1h", "1d")
        
    Returns:
        Interval in seconds
    """
    units = {
        's': 1,
        'm': 60,
        'h': 3600,
        'd': 86400,
        'w': 604800
    }
    
    value = int(interval[:-1])
    unit = interval[-1]
    
    return value * units.get(unit, 60)


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes to human-readable string
    
    Args:
        bytes_value: Number of bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    
    return f"{bytes_value:.1f} PB"


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "2h 30m")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    
    hours = minutes / 60
    if hours < 24:
        return f"{hours:.1f}h"
    
    days = hours / 24
    return f"{days:.1f}d"


def calculate_percentile(values: List[float], percentile: int) -> float:
    """
    Calculate percentile from list of values
    
    Args:
        values: List of numeric values
        percentile: Percentile to calculate (0-100)
        
    Returns:
        Percentile value
    """
    if not values:
        return 0.0
    
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * percentile / 100
    f = int(k)
    c = f + 1
    
    if c >= len(sorted_values):
        return sorted_values[-1]
    
    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    
    return d0 + d1


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple dictionaries
    
    Args:
        *dicts: Dictionaries to merge
        
    Returns:
        Merged dictionary
    """
    result = {}
    
    for d in dicts:
        result.update(d)
    
    return result


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division fails
        
    Returns:
        Result of division or default
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ValueError, ZeroDivisionError):
        return default


def retry_on_exception(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator to retry function on exception
    
    Args:
        max_retries: Maximum number of retries
        delay: Initial delay between retries
        backoff: Backoff multiplier
        exceptions: Tuple of exceptions to catch
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            current_delay = delay
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries - 1:
                        raise
                    
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            current_delay = delay
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries - 1:
                        raise
                    
                    time.sleep(current_delay)
                    current_delay *= backoff
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


def timeit(func):
    """
    Decorator to measure function execution time
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} took {elapsed:.3f}s")
        return result
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} took {elapsed:.3f}s")
        return result
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


class RateLimiter:
    """Simple rate limiter"""
    
    def __init__(self, max_calls: int, time_window: float):
        """
        Initialize rate limiter
        
        Args:
            max_calls: Maximum calls allowed in time window
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    def allow_request(self) -> bool:
        """
        Check if request is allowed
        
        Returns:
            True if request is allowed, False otherwise
        """
        now = time.time()
        
        # Remove old calls
        self.calls = [call for call in self.calls if now - call < self.time_window]
        
        # Check if limit reached
        if len(self.calls) >= self.max_calls:
            return False
        
        # Add current call
        self.calls.append(now)
        return True
    
    def reset(self):
        """Reset rate limiter"""
        self.calls = []


class CircularBuffer:
    """Simple circular buffer for storing recent items"""
    
    def __init__(self, max_size: int):
        """
        Initialize circular buffer
        
        Args:
            max_size: Maximum number of items to store
        """
        self.max_size = max_size
        self.buffer = []
        self.index = 0
    
    def append(self, item: Any):
        """Add item to buffer"""
        if len(self.buffer) < self.max_size:
            self.buffer.append(item)
        else:
            self.buffer[self.index] = item
            self.index = (self.index + 1) % self.max_size
    
    def get_all(self) -> List[Any]:
        """Get all items in chronological order"""
        if len(self.buffer) < self.max_size:
            return self.buffer.copy()
        
        return self.buffer[self.index:] + self.buffer[:self.index]
    
    def clear(self):
        """Clear buffer"""
        self.buffer = []
        self.index = 0


def get_time_ago_string(timestamp: datetime) -> str:
    """
    Get human-readable "time ago" string
    
    Args:
        timestamp: Timestamp to convert
        
    Returns:
        Human-readable string (e.g., "2 hours ago")
    """
    now = datetime.utcnow()
    diff = now - timestamp
    
    seconds = diff.total_seconds()
    
    if seconds < 60:
        return "just now"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    else:
        days = int(seconds / 86400)
        return f"{days} day{'s' if days != 1 else ''} ago"


def extract_error_message(exception: Exception) -> str:
    """
    Extract clean error message from exception
    
    Args:
        exception: Exception object
        
    Returns:
        Clean error message
    """
    error_msg = str(exception)
    
    # Remove common prefixes
    prefixes = ['Error: ', 'Exception: ', 'RuntimeError: ']
    for prefix in prefixes:
        if error_msg.startswith(prefix):
            error_msg = error_msg[len(prefix):]
    
    return error_msg.strip()