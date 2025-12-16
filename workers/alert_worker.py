"""
Alert Processing Worker - Monitors anomalies and triggers alerts
"""
import sys
import os
import time
import signal
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from pymongo import MongoClient
from pymongo.collection import Collection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [ALERT_WORKER] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AlertRule:
    """Alert rule definition"""
    
    def __init__(
        self,
        name: str,
        condition: str,
        threshold: float,
        severity: str,
        window_minutes: int = 5
    ):
        self.name = name
        self.condition = condition
        self.threshold = threshold
        self.severity = severity
        self.window_minutes = window_minutes


class AlertWorker:
    """Worker that monitors anomalies and generates alerts"""
    
    def __init__(
        self,
        mongo_uri: str = "mongodb://localhost:27017",
        db_name: str = "log_analytics",
        check_interval: int = 60
    ):
        """
        Initialize alert worker
        
        Args:
            mongo_uri: MongoDB connection URI
            db_name: Database name
            check_interval: Seconds between checks
        """
        self.mongo_client = MongoClient(mongo_uri)
        self.db = self.mongo_client[db_name]
        
        self.anomalies_collection = self.db['anomalies']
        self.alerts_collection = self.db['alerts']
        self.logs_collection = self.db['logs']
        
        self.check_interval = check_interval
        self.running = False
        
        # Alert rules
        self.rules = self._load_default_rules()
        
        # Alert cooldown to prevent spam
        self.alert_cooldown = {}  # rule_name -> last_alert_time
        self.cooldown_minutes = 5
        
        self.stats = {
            'checks_performed': 0,
            'alerts_generated': 0,
            'errors': 0,
            'started_at': None
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
    
    def _load_default_rules(self) -> List[AlertRule]:
        """Load default alert rules"""
        return [
            AlertRule(
                name="high_anomaly_rate",
                condition="anomaly_rate_exceeds",
                threshold=0.1,  # 10% anomaly rate
                severity="high",
                window_minutes=5
            ),
            AlertRule(
                name="critical_anomaly_detected",
                condition="critical_anomaly",
                threshold=0.8,  # Anomaly score >= 0.8
                severity="critical",
                window_minutes=1
            ),
            AlertRule(
                name="error_spike",
                condition="error_rate_spike",
                threshold=0.2,  # 20% error rate
                severity="high",
                window_minutes=5
            ),
            AlertRule(
                name="service_degradation",
                condition="slow_response_time",
                threshold=2000,  # 2 seconds
                severity="medium",
                window_minutes=10
            )
        ]
    
    def start(self) -> None:
        """Start the alert worker"""
        logger.info("Starting Alert Worker")
        self.running = True
        self.stats['started_at'] = datetime.utcnow().isoformat()
        
        while self.running:
            try:
                # Check all rules
                for rule in self.rules:
                    self._check_rule(rule)
                
                self.stats['checks_performed'] += 1
                
                if self.stats['checks_performed'] % 10 == 0:
                    logger.info(
                        f"Performed {self.stats['checks_performed']} checks, "
                        f"generated {self.stats['alerts_generated']} alerts"
                    )
                
                # Sleep until next check
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in alert loop: {str(e)}", exc_info=True)
                self.stats['errors'] += 1
                time.sleep(5)
        
        logger.info("Alert worker stopped")
    
    def _check_rule(self, rule: AlertRule) -> None:
        """
        Check if a rule is triggered
        
        Args:
            rule: AlertRule to check
        """
        # Check cooldown
        if not self._can_alert(rule.name):
            return
        
        try:
            if rule.condition == "anomaly_rate_exceeds":
                self._check_anomaly_rate(rule)
            elif rule.condition == "critical_anomaly":
                self._check_critical_anomalies(rule)
            elif rule.condition == "error_rate_spike":
                self._check_error_rate(rule)
            elif rule.condition == "slow_response_time":
                self._check_response_time(rule)
                
        except Exception as e:
            logger.error(f"Error checking rule {rule.name}: {str(e)}")
    
    def _check_anomaly_rate(self, rule: AlertRule) -> None:
        """Check if anomaly rate exceeds threshold"""
        window_start = datetime.utcnow() - timedelta(minutes=rule.window_minutes)
        
        # Count recent logs
        total_logs = self.logs_collection.count_documents({
            'timestamp': {'$gte': window_start}
        })
        
        if total_logs == 0:
            return
        
        # Count recent anomalies
        anomalies = self.anomalies_collection.count_documents({
            'detected_at': {'$gte': window_start}
        })
        
        anomaly_rate = anomalies / total_logs if total_logs > 0 else 0
        
        if anomaly_rate >= rule.threshold:
            self._trigger_alert(
                rule=rule,
                title=f"High Anomaly Rate: {anomaly_rate:.1%}",
                description=(
                    f"Anomaly rate of {anomaly_rate:.1%} exceeds threshold "
                    f"of {rule.threshold:.1%} in the last {rule.window_minutes} minutes"
                ),
                metrics={
                    'anomaly_rate': anomaly_rate,
                    'total_logs': total_logs,
                    'anomalies': anomalies,
                    'window_minutes': rule.window_minutes
                }
            )
    
    def _check_critical_anomalies(self, rule: AlertRule) -> None:
        """Check for critical anomalies"""
        window_start = datetime.utcnow() - timedelta(minutes=rule.window_minutes)
        
        critical_anomalies = list(self.anomalies_collection.find({
            'detected_at': {'$gte': window_start},
            'anomaly_score': {'$gte': rule.threshold},
            'status': 'new'
        }).limit(10))
        
        if critical_anomalies:
            self._trigger_alert(
                rule=rule,
                title=f"Critical Anomalies Detected ({len(critical_anomalies)})",
                description=(
                    f"Found {len(critical_anomalies)} critical anomalies "
                    f"with scores >= {rule.threshold}"
                ),
                metrics={
                    'count': len(critical_anomalies),
                    'highest_score': max(a['anomaly_score'] for a in critical_anomalies),
                    'services': list(set(a.get('service') for a in critical_anomalies))
                },
                related_anomalies=[str(a['_id']) for a in critical_anomalies]
            )
    
    def _check_error_rate(self, rule: AlertRule) -> None:
        """Check for error rate spikes"""
        window_start = datetime.utcnow() - timedelta(minutes=rule.window_minutes)
        
        # Count total requests
        total_logs = self.logs_collection.count_documents({
            'timestamp': {'$gte': window_start},
            'status_code': {'$exists': True}
        })
        
        if total_logs == 0:
            return
        
        # Count errors (status >= 400)
        errors = self.logs_collection.count_documents({
            'timestamp': {'$gte': window_start},
            'status_code': {'$gte': 400}
        })
        
        error_rate = errors / total_logs if total_logs > 0 else 0
        
        if error_rate >= rule.threshold:
            self._trigger_alert(
                rule=rule,
                title=f"Error Rate Spike: {error_rate:.1%}",
                description=(
                    f"Error rate of {error_rate:.1%} exceeds threshold "
                    f"of {rule.threshold:.1%}"
                ),
                metrics={
                    'error_rate': error_rate,
                    'total_requests': total_logs,
                    'errors': errors
                }
            )
    
    def _check_response_time(self, rule: AlertRule) -> None:
        """Check for slow response times"""
        window_start = datetime.utcnow() - timedelta(minutes=rule.window_minutes)
        
        # Get recent logs with response times
        pipeline = [
            {
                '$match': {
                    'timestamp': {'$gte': window_start},
                    'response_time': {'$exists': True}
                }
            },
            {
                '$group': {
                    '_id': None,
                    'avg_response': {'$avg': '$response_time'},
                    'max_response': {'$max': '$response_time'},
                    'count': {'$sum': 1}
                }
            }
        ]
        
        result = list(self.logs_collection.aggregate(pipeline))
        
        if not result:
            return
        
        stats = result[0]
        avg_response = stats['avg_response']
        
        if avg_response >= rule.threshold:
            self._trigger_alert(
                rule=rule,
                title=f"Slow Response Times: {avg_response:.0f}ms",
                description=(
                    f"Average response time of {avg_response:.0f}ms "
                    f"exceeds threshold of {rule.threshold}ms"
                ),
                metrics={
                    'avg_response_time': avg_response,
                    'max_response_time': stats['max_response'],
                    'sample_size': stats['count']
                }
            )
    
    def _trigger_alert(
        self,
        rule: AlertRule,
        title: str,
        description: str,
        metrics: Dict,
        related_anomalies: Optional[List[str]] = None
    ) -> None:
        """
        Trigger an alert
        
        Args:
            rule: Alert rule that triggered
            title: Alert title
            description: Alert description
            metrics: Related metrics
            related_anomalies: IDs of related anomalies
        """
        alert_doc = {
            'rule_name': rule.name,
            'title': title,
            'description': description,
            'severity': rule.severity,
            'metrics': metrics,
            'related_anomalies': related_anomalies or [],
            'triggered_at': datetime.utcnow(),
            'status': 'active',
            'acknowledged': False,
            'resolved': False
        }
        
        try:
            self.alerts_collection.insert_one(alert_doc)
            self.stats['alerts_generated'] += 1
            
            # Update cooldown
            self.alert_cooldown[rule.name] = datetime.utcnow()
            
            logger.warning(f"ALERT TRIGGERED: {title}")
            logger.warning(f"  Severity: {rule.severity}")
            logger.warning(f"  Description: {description}")
            
        except Exception as e:
            logger.error(f"Failed to store alert: {str(e)}")
    
    def _can_alert(self, rule_name: str) -> bool:
        """Check if we can alert for this rule (cooldown check)"""
        if rule_name not in self.alert_cooldown:
            return True
        
        last_alert = self.alert_cooldown[rule_name]
        cooldown_end = last_alert + timedelta(minutes=self.cooldown_minutes)
        
        return datetime.utcnow() >= cooldown_end
    
    def _handle_shutdown(self, signum, frame) -> None:
        """Handle shutdown signals"""
        logger.info(f"Received shutdown signal: {signum}")
        self.running = False
    
    def get_stats(self) -> Dict:
        """Get worker statistics"""
        return {
            **self.stats,
            'active_rules': len(self.rules),
            'running': self.running,
            'current_time': datetime.utcnow().isoformat()
        }
    
    def stop(self) -> None:
        """Stop the worker"""
        self.running = False
        logger.info("Stopping alert worker...")


def main():
    """Main worker entry point"""
    logger.info("Starting Alert Processing Worker")
    
    # Configuration
    mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
    db_name = os.getenv('DB_NAME', 'log_analytics')
    check_interval = int(os.getenv('ALERT_CHECK_INTERVAL', '60'))
    
    # Create worker
    worker = AlertWorker(
        mongo_uri=mongo_uri,
        db_name=db_name,
        check_interval=check_interval
    )
    
    try:
        # Start monitoring
        worker.start()
    except Exception as e:
        logger.error(f"Worker failed: {str(e)}", exc_info=True)
    finally:
        stats = worker.get_stats()
        logger.info("=" * 50)
        logger.info("Worker Statistics:")
        logger.info(f"  Checks: {stats['checks_performed']}")
        logger.info(f"  Alerts: {stats['alerts_generated']}")
        logger.info(f"  Errors: {stats['errors']}")
        logger.info("=" * 50)


if __name__ == "__main__":
    main()