"""
Input validation utilities
"""
import re
from datetime import datetime, timedelta
from typing import Any, Optional, List, Dict
from ipaddress import ip_address, ip_network, AddressValueError


class ValidationError(Exception):
    """Custom validation error"""
    pass


class LogValidator:
    """Validator for log entries"""
    
    VALID_LOG_LEVELS = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', 'FATAL'}
    MAX_MESSAGE_LENGTH = 10000
    MAX_SERVICE_NAME_LENGTH = 100
    
    @staticmethod
    def validate_log_level(level: str) -> str:
        """
        Validate log level
        
        Args:
            level: Log level string
            
        Returns:
            Normalized log level
            
        Raises:
            ValidationError: If level is invalid
        """
        level = level.upper().strip()
        
        if level not in LogValidator.VALID_LOG_LEVELS:
            raise ValidationError(
                f"Invalid log level: {level}. "
                f"Must be one of: {', '.join(LogValidator.VALID_LOG_LEVELS)}"
            )
        
        return level
    
    @staticmethod
    def validate_message(message: str) -> str:
        """
        Validate log message
        
        Args:
            message: Log message
            
        Returns:
            Validated message
            
        Raises:
            ValidationError: If message is invalid
        """
        if not message or not isinstance(message, str):
            raise ValidationError("Log message must be a non-empty string")
        
        message = message.strip()
        
        if len(message) > LogValidator.MAX_MESSAGE_LENGTH:
            raise ValidationError(
                f"Log message too long. Maximum length: {LogValidator.MAX_MESSAGE_LENGTH}"
            )
        
        return message
    
    @staticmethod
    def validate_service_name(service: str) -> str:
        """
        Validate service name
        
        Args:
            service: Service name
            
        Returns:
            Validated service name
            
        Raises:
            ValidationError: If service name is invalid
        """
        if not service or not isinstance(service, str):
            raise ValidationError("Service name must be a non-empty string")
        
        service = service.strip()
        
        if len(service) > LogValidator.MAX_SERVICE_NAME_LENGTH:
            raise ValidationError(
                f"Service name too long. Maximum length: {LogValidator.MAX_SERVICE_NAME_LENGTH}"
            )
        
        # Check for valid characters (alphanumeric, dash, underscore)
        if not re.match(r'^[a-zA-Z0-9_-]+$', service):
            raise ValidationError(
                "Service name must contain only alphanumeric characters, dashes, and underscores"
            )
        
        return service
    
    @staticmethod
    def validate_timestamp(timestamp: Any) -> datetime:
        """
        Validate and convert timestamp
        
        Args:
            timestamp: Timestamp (datetime, string, or int)
            
        Returns:
            Validated datetime object
            
        Raises:
            ValidationError: If timestamp is invalid
        """
        if isinstance(timestamp, datetime):
            return timestamp
        
        if isinstance(timestamp, str):
            try:
                # Try ISO format
                return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except ValueError:
                raise ValidationError(f"Invalid timestamp format: {timestamp}")
        
        if isinstance(timestamp, (int, float)):
            try:
                # Assume Unix timestamp
                return datetime.fromtimestamp(timestamp)
            except (ValueError, OSError):
                raise ValidationError(f"Invalid Unix timestamp: {timestamp}")
        
        raise ValidationError(f"Invalid timestamp type: {type(timestamp)}")
    
    @staticmethod
    def validate_response_time(response_time: Any) -> Optional[float]:
        """
        Validate response time
        
        Args:
            response_time: Response time value
            
        Returns:
            Validated response time or None
            
        Raises:
            ValidationError: If response time is invalid
        """
        if response_time is None:
            return None
        
        try:
            rt = float(response_time)
            
            if rt < 0:
                raise ValidationError("Response time cannot be negative")
            
            if rt > 3600000:  # 1 hour in milliseconds
                raise ValidationError("Response time seems unreasonably high")
            
            return rt
            
        except (ValueError, TypeError):
            raise ValidationError(f"Invalid response time value: {response_time}")


class TimeRangeValidator:
    """Validator for time ranges"""
    
    MAX_TIME_RANGE_DAYS = 90
    
    @staticmethod
    def validate_time_range(
        start_time: datetime,
        end_time: datetime,
        max_days: Optional[int] = None
    ) -> tuple[datetime, datetime]:
        """
        Validate time range
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            max_days: Maximum allowed range in days
            
        Returns:
            Validated (start_time, end_time) tuple
            
        Raises:
            ValidationError: If time range is invalid
        """
        if start_time >= end_time:
            raise ValidationError("Start time must be before end time")
        
        max_days = max_days or TimeRangeValidator.MAX_TIME_RANGE_DAYS
        max_delta = timedelta(days=max_days)
        
        if (end_time - start_time) > max_delta:
            raise ValidationError(
                f"Time range too large. Maximum: {max_days} days"
            )
        
        # Check if times are too far in the future
        now = datetime.utcnow()
        if start_time > now + timedelta(days=1):
            raise ValidationError("Start time cannot be more than 1 day in the future")
        
        return start_time, end_time


class IPValidator:
    """Validator for IP addresses"""
    
    @staticmethod
    def validate_ip_address(ip: str) -> str:
        """
        Validate IP address (IPv4 or IPv6)
        
        Args:
            ip: IP address string
            
        Returns:
            Validated IP address
            
        Raises:
            ValidationError: If IP is invalid
        """
        try:
            ip_address(ip)
            return ip
        except AddressValueError:
            raise ValidationError(f"Invalid IP address: {ip}")
    
    @staticmethod
    def validate_ip_network(network: str) -> str:
        """
        Validate IP network (CIDR notation)
        
        Args:
            network: IP network string
            
        Returns:
            Validated IP network
            
        Raises:
            ValidationError: If network is invalid
        """
        try:
            ip_network(network)
            return network
        except (AddressValueError, ValueError):
            raise ValidationError(f"Invalid IP network: {network}")


class QueryValidator:
    """Validator for query parameters"""
    
    @staticmethod
    def validate_limit(limit: int, max_limit: int = 1000) -> int:
        """
        Validate limit parameter
        
        Args:
            limit: Requested limit
            max_limit: Maximum allowed limit
            
        Returns:
            Validated limit
            
        Raises:
            ValidationError: If limit is invalid
        """
        if limit < 1:
            raise ValidationError("Limit must be at least 1")
        
        if limit > max_limit:
            raise ValidationError(f"Limit too large. Maximum: {max_limit}")
        
        return limit
    
    @staticmethod
    def validate_offset(offset: int) -> int:
        """
        Validate offset parameter
        
        Args:
            offset: Requested offset
            
        Returns:
            Validated offset
            
        Raises:
            ValidationError: If offset is invalid
        """
        if offset < 0:
            raise ValidationError("Offset cannot be negative")
        
        return offset
    
    @staticmethod
    def validate_interval(interval: str) -> str:
        """
        Validate time interval string
        
        Args:
            interval: Interval string (e.g., "5m", "1h", "1d")
            
        Returns:
            Validated interval
            
        Raises:
            ValidationError: If interval is invalid
        """
        pattern = r'^(\d+)([smhd])$'
        match = re.match(pattern, interval)
        
        if not match:
            raise ValidationError(
                "Invalid interval format. Use: <number><unit> where unit is s, m, h, or d"
            )
        
        value, unit = match.groups()
        value = int(value)
        
        # Set reasonable limits
        limits = {
            's': 3600,  # Max 1 hour
            'm': 1440,  # Max 24 hours
            'h': 720,   # Max 30 days
            'd': 365    # Max 1 year
        }
        
        if value > limits[unit]:
            raise ValidationError(
                f"Interval too large. Maximum for {unit}: {limits[unit]}"
            )
        
        return interval


class AlertValidator:
    """Validator for alert-related data"""
    
    VALID_SEVERITIES = {'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
    VALID_STATUSES = {'OPEN', 'ACKNOWLEDGED', 'RESOLVED'}
    
    @staticmethod
    def validate_severity(severity: str) -> str:
        """Validate alert severity"""
        severity = severity.upper().strip()
        
        if severity not in AlertValidator.VALID_SEVERITIES:
            raise ValidationError(
                f"Invalid severity: {severity}. "
                f"Must be one of: {', '.join(AlertValidator.VALID_SEVERITIES)}"
            )
        
        return severity
    
    @staticmethod
    def validate_status(status: str) -> str:
        """Validate alert status"""
        status = status.upper().strip()
        
        if status not in AlertValidator.VALID_STATUSES:
            raise ValidationError(
                f"Invalid status: {status}. "
                f"Must be one of: {', '.join(AlertValidator.VALID_STATUSES)}"
            )
        
        return status
    
    @staticmethod
    def validate_threshold(threshold: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate alert threshold configuration
        
        Args:
            threshold: Threshold configuration
            
        Returns:
            Validated threshold
            
        Raises:
            ValidationError: If threshold is invalid
        """
        required_fields = ['metric', 'operator', 'value']
        
        for field in required_fields:
            if field not in threshold:
                raise ValidationError(f"Missing required field in threshold: {field}")
        
        # Validate operator
        valid_operators = {'gt', 'gte', 'lt', 'lte', 'eq', 'ne'}
        if threshold['operator'] not in valid_operators:
            raise ValidationError(
                f"Invalid operator: {threshold['operator']}. "
                f"Must be one of: {', '.join(valid_operators)}"
            )
        
        # Validate value is numeric
        try:
            float(threshold['value'])
        except (ValueError, TypeError):
            raise ValidationError(f"Threshold value must be numeric: {threshold['value']}")
        
        return threshold


def sanitize_string(value: str, max_length: Optional[int] = None) -> str:
    """
    Sanitize string input
    
    Args:
        value: String to sanitize
        max_length: Optional maximum length
        
    Returns:
        Sanitized string
    """
    if not isinstance(value, str):
        value = str(value)
    
    # Remove null bytes
    value = value.replace('\x00', '')
    
    # Strip whitespace
    value = value.strip()
    
    # Truncate if needed
    if max_length and len(value) > max_length:
        value = value[:max_length]
    
    return value


def validate_json_structure(data: Dict[str, Any], required_fields: List[str]) -> None:
    """
    Validate JSON structure has required fields
    
    Args:
        data: Dictionary to validate
        required_fields: List of required field names
        
    Raises:
        ValidationError: If required fields are missing
    """
    missing_fields = [field for field in required_fields if field not in data]
    
    if missing_fields:
        raise ValidationError(
            f"Missing required fields: {', '.join(missing_fields)}"
        )