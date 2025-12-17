"""
Log parsing utilities for various log formats
"""
import re
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum


class LogFormat(str, Enum):
    """Supported log formats"""
    JSON = "json"
    APACHE = "apache"
    NGINX = "nginx"
    SYSLOG = "syslog"
    COMMON = "common"
    COMBINED = "combined"


class LogParser:
    """Parser for various log formats"""
    
    # Common log patterns
    APACHE_COMMON = r'(?P<host>[\d\.]+) - - \[(?P<timestamp>[^\]]+)\] "(?P<method>\w+) (?P<path>[^\s]+) HTTP/[\d\.]+" (?P<status>\d+) (?P<size>\d+|-)'
    
    APACHE_COMBINED = r'(?P<host>[\d\.]+) - - \[(?P<timestamp>[^\]]+)\] "(?P<method>\w+) (?P<path>[^\s]+) HTTP/[\d\.]+" (?P<status>\d+) (?P<size>\d+|-) "(?P<referrer>[^"]*)" "(?P<user_agent>[^"]*)"'
    
    NGINX = r'(?P<host>[\d\.]+) - - \[(?P<timestamp>[^\]]+)\] "(?P<method>\w+) (?P<path>[^\s]+) HTTP/[\d\.]+" (?P<status>\d+) (?P<size>\d+) "(?P<referrer>[^"]*)" "(?P<user_agent>[^"]*)" "(?P<response_time>[\d\.]+)"'
    
    SYSLOG = r'(?P<timestamp>\w+\s+\d+\s+\d+:\d+:\d+) (?P<host>\S+) (?P<process>\S+)(?:\[(?P<pid>\d+)\])?: (?P<message>.*)'
    
    def __init__(self):
        self.patterns = {
            LogFormat.APACHE: re.compile(self.APACHE_COMMON),
            LogFormat.COMBINED: re.compile(self.APACHE_COMBINED),
            LogFormat.NGINX: re.compile(self.NGINX),
            LogFormat.SYSLOG: re.compile(self.SYSLOG),
        }
    
    def parse(self, log_line: str, format_type: Optional[LogFormat] = None) -> Optional[Dict[str, Any]]:
        """
        Parse a log line
        
        Args:
            log_line: Raw log line
            format_type: Specific format to try, or None to auto-detect
            
        Returns:
            Parsed log dictionary or None
        """
        log_line = log_line.strip()
        
        if not log_line:
            return None
        
        # Try JSON first
        if log_line.startswith('{'):
            return self._parse_json(log_line)
        
        # Try specific format
        if format_type and format_type in self.patterns:
            result = self._parse_with_pattern(log_line, self.patterns[format_type])
            if result:
                return result
        
        # Auto-detect format
        for fmt, pattern in self.patterns.items():
            result = self._parse_with_pattern(log_line, pattern)
            if result:
                result['_format'] = fmt.value
                return result
        
        # Fallback to unstructured
        return self._parse_unstructured(log_line)
    
    def parse_batch(self, log_lines: List[str], format_type: Optional[LogFormat] = None) -> List[Dict[str, Any]]:
        """
        Parse multiple log lines
        
        Args:
            log_lines: List of raw log lines
            format_type: Specific format to try
            
        Returns:
            List of parsed log dictionaries
        """
        parsed_logs = []
        
        for line in log_lines:
            parsed = self.parse(line, format_type)
            if parsed:
                parsed_logs.append(parsed)
        
        return parsed_logs
    
    def _parse_json(self, log_line: str) -> Optional[Dict[str, Any]]:
        """Parse JSON formatted log"""
        try:
            data = json.loads(log_line)
            
            # Normalize common fields
            normalized = {
                'message': data.get('message') or data.get('msg') or str(data),
                'level': self._normalize_level(
                    data.get('level') or data.get('severity') or 'INFO'
                ),
                'timestamp': self._parse_timestamp(
                    data.get('timestamp') or data.get('time') or data.get('@timestamp')
                ),
                'service': data.get('service') or data.get('app') or 'unknown',
                '_format': 'json'
            }
            
            # Copy additional fields
            for key, value in data.items():
                if key not in normalized:
                    normalized[key] = value
            
            return normalized
            
        except json.JSONDecodeError:
            return None
    
    def _parse_with_pattern(self, log_line: str, pattern: re.Pattern) -> Optional[Dict[str, Any]]:
        """Parse log line with regex pattern"""
        match = pattern.match(log_line)
        
        if not match:
            return None
        
        data = match.groupdict()
        
        # Normalize common fields
        normalized = {
            'message': data.get('message', log_line),
            'timestamp': self._parse_timestamp(data.get('timestamp')),
            'host': data.get('host', 'unknown'),
        }
        
        # Add HTTP-specific fields
        if 'method' in data:
            normalized['http_method'] = data['method']
            normalized['http_path'] = data.get('path', '')
            normalized['http_status'] = int(data.get('status', 0))
            
            # Determine log level based on status code
            status = normalized['http_status']
            if status >= 500:
                normalized['level'] = 'ERROR'
            elif status >= 400:
                normalized['level'] = 'WARNING'
            else:
                normalized['level'] = 'INFO'
        else:
            normalized['level'] = 'INFO'
        
        # Add response time if available
        if 'response_time' in data:
            try:
                normalized['response_time'] = float(data['response_time'])
            except (ValueError, TypeError):
                pass
        
        # Add other fields
        for key, value in data.items():
            if key not in normalized and value and value != '-':
                normalized[key] = value
        
        return normalized
    
    def _parse_unstructured(self, log_line: str) -> Dict[str, Any]:
        """Parse unstructured log line"""
        # Try to extract timestamp from beginning
        timestamp_pattern = r'^(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)'
        timestamp_match = re.match(timestamp_pattern, log_line)
        
        timestamp = None
        if timestamp_match:
            timestamp = self._parse_timestamp(timestamp_match.group(1))
        
        # Try to detect log level
        level = 'INFO'
        level_pattern = r'\b(DEBUG|INFO|WARN(?:ING)?|ERROR|CRITICAL|FATAL)\b'
        level_match = re.search(level_pattern, log_line, re.IGNORECASE)
        if level_match:
            level = self._normalize_level(level_match.group(1))
        
        return {
            'message': log_line,
            'level': level,
            'timestamp': timestamp or datetime.utcnow(),
            '_format': 'unstructured'
        }
    
    def _parse_timestamp(self, timestamp_str: Optional[str]) -> Optional[datetime]:
        """Parse various timestamp formats"""
        if not timestamp_str:
            return None
        
        # Common timestamp formats
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%S%z',
            '%Y-%m-%dT%H:%M:%S.%f%z',
            '%d/%b/%Y:%H:%M:%S %z',
            '%b %d %H:%M:%S',
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str.strip(), fmt)
            except (ValueError, AttributeError):
                continue
        
        # Try ISO format
        try:
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            pass
        
        return None
    
    def _normalize_level(self, level: str) -> str:
        """Normalize log level to standard values"""
        level = level.upper()
        
        if level in ['WARN', 'WARNING']:
            return 'WARNING'
        elif level in ['ERR', 'ERROR']:
            return 'ERROR'
        elif level in ['CRIT', 'CRITICAL', 'FATAL']:
            return 'CRITICAL'
        elif level == 'DEBUG':
            return 'DEBUG'
        else:
            return 'INFO'
    
    @staticmethod
    def extract_metadata(log_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from parsed log"""
        metadata = {}
        
        # Extract IP address
        if 'host' in log_data:
            metadata['ip_address'] = log_data['host']
        
        # Extract user agent info
        if 'user_agent' in log_data:
            metadata['user_agent'] = log_data['user_agent']
        
        # Extract HTTP info
        if 'http_method' in log_data:
            metadata['http'] = {
                'method': log_data.get('http_method'),
                'path': log_data.get('http_path'),
                'status': log_data.get('http_status'),
                'referrer': log_data.get('referrer')
            }
        
        # Extract process info
        if 'process' in log_data:
            metadata['process'] = log_data['process']
            if 'pid' in log_data:
                metadata['pid'] = log_data['pid']
        
        return metadata


# Singleton instance
parser = LogParser()


def parse_log_line(log_line: str, format_type: Optional[LogFormat] = None) -> Optional[Dict[str, Any]]:
    """
    Convenience function to parse a single log line
    
    Args:
        log_line: Raw log line
        format_type: Optional specific format
        
    Returns:
        Parsed log dictionary
    """
    return parser.parse(log_line, format_type)


def parse_log_batch(log_lines: List[str], format_type: Optional[LogFormat] = None) -> List[Dict[str, Any]]:
    """
    Convenience function to parse multiple log lines
    
    Args:
        log_lines: List of raw log lines
        format_type: Optional specific format
        
    Returns:
        List of parsed log dictionaries
    """
    return parser.parse_batch(log_lines, format_type)