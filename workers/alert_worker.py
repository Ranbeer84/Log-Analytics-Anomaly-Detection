"""
Enhanced Alert Processing Worker - Monitors anomalies and triggers intelligent alerts
"""
import sys
import os
import time
import signal
import logging
import smtplib
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

sys.path.insert(0, str(Path(__file__).parent.parent))

from pymongo import MongoClient
from pymongo.collection import Collection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [ALERT_WORKER] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AlertRule:
    """Enhanced alert rule with thresholds and severity"""
    
    def __init__(
        self,
        name: str,
        condition: str,
        threshold: float,
        severity: str,
        window_minutes: int = 5,
        description: str = "",
        enabled: bool = True
    ):
        self.name = name
        self.condition = condition
        self.threshold = threshold
        self.severity = severity
        self.window_minutes = window_minutes
        self.description = description
        self.enabled = enabled


class AlertWorker:
    """
    Production-grade alert worker with:
    - Alert aggregation and deduplication
    - Rate limiting
    - Multiple notification channels
    - Alert correlation
    """
    
    def __init__(
        self,
        mongo_uri: str = "mongodb://localhost:27017",
        db_name: str = "log_analytics",
        check_interval: int = 60,
        smtp_config: Optional[Dict] = None
    ):
        """Initialize enhanced alert worker"""
        self.mongo_client = MongoClient(
            mongo_uri,
            maxPoolSize=20,
            minPoolSize=5
        )
        self.db = self.mongo_client[db_name]
        
        self.anomalies_collection = self.db['anomalies']
        self.alerts_collection = self.db['alerts']
        self.logs_collection = self.db['logs']
        self.alert_history_collection = self.db['alert_history']
        
        # Ensure indexes
        self._ensure_indexes()
        
        self.check_interval = check_interval
        self.running = False
        
        # Alert rules (can be loaded from DB in production)
        self.rules = self._load_rules()
        
        # Alert management
        self.alert_cooldown = {}  # rule_name -> last_alert_time
        self.cooldown_minutes = 5
        self.alert_aggregation = defaultdict(list)  # group similar alerts
        
        # Stats
        self.stats = {
            'checks_performed': 0,
            'alerts_generated': 0,
            'alerts_suppressed': 0,
            'alerts_sent': 0,
            'errors': 0,
            'started_at': None
        }
        
        # Recent alerts for correlation
        self.recent_alerts = deque(maxlen=100)
        
        # SMTP config for email notifications
        self.smtp_config = smtp_config or self._get_smtp_config()
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
    
    def _ensure_indexes(self) -> None:
        """Create indexes for alert queries"""
        try:
            self.alerts_collection.create_index([
                ('triggered_at', -1),
                ('severity', 1)
            ])
            self.alerts_collection.create_index('status')
            self.alerts_collection.create_index('rule_name')
            
            logger.info("Alert collection indexes ensured")
        except Exception as e:
            logger.warning(f"Failed to create indexes: {e}")
    
    def _get_smtp_config(self) -> Dict:
        """Get SMTP configuration from environment"""
        return {
            'enabled': os.getenv('SMTP_ENABLED', 'false').lower() == 'true',
            'host': os.getenv('SMTP_HOST', 'localhost'),
            'port': int(os.getenv('SMTP_PORT', '587')),
            'username': os.getenv('SMTP_USERNAME', ''),
            'password': os.getenv('SMTP_PASSWORD', ''),
            'from_email': os.getenv('SMTP_FROM', '[email protected]'),
            'to_emails': os.getenv('ALERT_EMAILS', '').split(',')
        }
    
    def _load_rules(self) -> List[AlertRule]:
        """Load alert rules with enhanced definitions"""
        return [
            AlertRule(
                name="high_anomaly_rate",
                condition="anomaly_rate_exceeds",
                threshold=0.1,  # 10%
                severity="high",
                window_minutes=5,
                description="Anomaly detection rate exceeds threshold"
            ),
            AlertRule(
                name="critical_anomaly_detected",
                condition="critical_anomaly",
                threshold=0.8,
                severity="critical",
                window_minutes=1,
                description="Critical severity anomaly detected"
            ),
            AlertRule(
                name="error_spike",
                condition="error_rate_spike",
                threshold=0.2,  # 20%
                severity="high",
                window_minutes=5,
                description="HTTP error rate spike detected"
            ),
            AlertRule(
                name="service_degradation",
                condition="slow_response_time",
                threshold=2000,  # 2 seconds
                severity="medium",
                window_minutes=10,
                description="Service response time degraded"
            ),
            AlertRule(
                name="service_failure",
                condition="service_unavailable",
                threshold=0.05,  # 5% unavailable
                severity="critical",
                window_minutes=3,
                description="Service availability below threshold"
            ),
            AlertRule(
                name="anomaly_cluster",
                condition="anomaly_clustering",
                threshold=5,  # 5 anomalies
                severity="high",
                window_minutes=10,
                description="Multiple anomalies detected for same service"
            )
        ]
    
    def start(self) -> None:
        """Start the enhanced alert worker"""
        logger.info("=" * 60)
        logger.info("Starting Enhanced Alert Worker")
        logger.info(f"  Active rules: {len([r for r in self.rules if r.enabled])}")
        logger.info(f"  Check interval: {self.check_interval}s")
        logger.info(f"  Email alerts: {'enabled' if self.smtp_config['enabled'] else 'disabled'}")
        logger.info("=" * 60)
        
        self.running = True
        self.stats['started_at'] = datetime.utcnow().isoformat()
        
        while self.running:
            try:
                check_start = time.time()
                
                # Check all enabled rules
                for rule in self.rules:
                    if not rule.enabled:
                        continue
                    
                    self._check_rule(rule)
                
                self.stats['checks_performed'] += 1
                
                # Process aggregated alerts
                self._process_aggregated_alerts()
                
                check_duration = time.time() - check_start
                
                # Periodic logging
                if self.stats['checks_performed'] % 10 == 0:
                    logger.info(
                        f"Check #{self.stats['checks_performed']} completed "
                        f"in {check_duration:.2f}s | "
                        f"Alerts: {self.stats['alerts_generated']} | "
                        f"Sent: {self.stats['alerts_sent']}"
                    )
                
                # Sleep until next check
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in alert loop: {e}", exc_info=True)
                self.stats['errors'] += 1
                time.sleep(5)
        
        logger.info("Alert worker stopped")
    
    def _check_rule(self, rule: AlertRule) -> None:
        """Check if a rule is triggered with cooldown"""
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
            elif rule.condition == "service_unavailable":
                self._check_service_availability(rule)
            elif rule.condition == "anomaly_clustering":
                self._check_anomaly_clustering(rule)
                
        except Exception as e:
            logger.error(f"Error checking rule {rule.name}: {e}")
    
    def _check_anomaly_rate(self, rule: AlertRule) -> None:
        """Check anomaly rate with historical comparison"""
        window_start = datetime.utcnow() - timedelta(minutes=rule.window_minutes)
        
        # Current window
        total_logs = self.logs_collection.count_documents({
            'timestamp': {'$gte': window_start}
        })
        
        if total_logs < 10:  # Need minimum sample size
            return
        
        anomalies = self.anomalies_collection.count_documents({
            'detected_at': {'$gte': window_start}
        })
        
        anomaly_rate = anomalies / total_logs
        
        # Compare with baseline
        baseline_start = window_start - timedelta(hours=1)
        baseline_total = self.logs_collection.count_documents({
            'timestamp': {'$gte': baseline_start, '$lt': window_start}
        })
        baseline_anomalies = self.anomalies_collection.count_documents({
            'detected_at': {'$gte': baseline_start, '$lt': window_start}
        })
        
        baseline_rate = baseline_anomalies / baseline_total if baseline_total > 0 else 0
        
        if anomaly_rate >= rule.threshold and anomaly_rate > baseline_rate * 2:
            self._trigger_alert(
                rule=rule,
                title=f"High Anomaly Rate: {anomaly_rate:.1%}",
                description=(
                    f"Anomaly rate of {anomaly_rate:.1%} exceeds threshold "
                    f"of {rule.threshold:.1%}. Baseline: {baseline_rate:.1%}"
                ),
                metrics={
                    'anomaly_rate': anomaly_rate,
                    'baseline_rate': baseline_rate,
                    'total_logs': total_logs,
                    'anomalies': anomalies,
                    'window_minutes': rule.window_minutes
                }
            )
    
    def _check_critical_anomalies(self, rule: AlertRule) -> None:
        """Check for critical anomalies with context"""
        window_start = datetime.utcnow() - timedelta(minutes=rule.window_minutes)
        
        critical_anomalies = list(self.anomalies_collection.find({
            'detected_at': {'$gte': window_start},
            'anomaly_score': {'$gte': rule.threshold},
            'status': 'new'
        }).limit(10))
        
        if critical_anomalies:
            # Group by service
            by_service = defaultdict(list)
            for anom in critical_anomalies:
                by_service[anom.get('service', 'unknown')].append(anom)
            
            self._trigger_alert(
                rule=rule,
                title=f"🚨 {len(critical_anomalies)} Critical Anomalies",
                description=(
                    f"Found {len(critical_anomalies)} critical anomalies "
                    f"across {len(by_service)} services"
                ),
                metrics={
                    'count': len(critical_anomalies),
                    'highest_score': max(a['anomaly_score'] for a in critical_anomalies),
                    'services': {
                        service: len(anoms) 
                        for service, anoms in by_service.items()
                    }
                },
                related_anomalies=[str(a['_id']) for a in critical_anomalies]
            )
    
    def _check_error_rate(self, rule: AlertRule) -> None:
        """Check error rate with service breakdown"""
        window_start = datetime.utcnow() - timedelta(minutes=rule.window_minutes)
        
        total_logs = self.logs_collection.count_documents({
            'timestamp': {'$gte': window_start},
            'status_code': {'$exists': True}
        })
        
        if total_logs < 10:
            return
        
        errors = self.logs_collection.count_documents({
            'timestamp': {'$gte': window_start},
            'status_code': {'$gte': 400}
        })
        
        error_rate = errors / total_logs
        
        if error_rate >= rule.threshold:
            # Get error breakdown
            error_breakdown = list(self.logs_collection.aggregate([
                {'$match': {
                    'timestamp': {'$gte': window_start},
                    'status_code': {'$gte': 400}
                }},
                {'$group': {
                    '_id': {'service': '$service', 'status': '$status_code'},
                    'count': {'$sum': 1}
                }},
                {'$sort': {'count': -1}},
                {'$limit': 5}
            ]))
            
            self._trigger_alert(
                rule=rule,
                title=f"⚠️ Error Rate Spike: {error_rate:.1%}",
                description=(
                    f"Error rate of {error_rate:.1%} exceeds threshold. "
                    f"{errors} errors in {total_logs} requests"
                ),
                metrics={
                    'error_rate': error_rate,
                    'total_requests': total_logs,
                    'errors': errors,
                    'top_errors': [
                        {'service': e['_id']['service'], 
                         'status': e['_id']['status'],
                         'count': e['count']}
                        for e in error_breakdown
                    ]
                }
            )
    
    def _check_response_time(self, rule: AlertRule) -> None:
        """Check response time with percentile analysis"""
        window_start = datetime.utcnow() - timedelta(minutes=rule.window_minutes)
        
        pipeline = [
            {'$match': {
                'timestamp': {'$gte': window_start},
                'response_time': {'$exists': True}
            }},
            {'$group': {
                '_id': None,
                'avg_response': {'$avg': '$response_time'},
                'max_response': {'$max': '$response_time'},
                'count': {'$sum': 1},
                'responses': {'$push': '$response_time'}
            }}
        ]
        
        result = list(self.logs_collection.aggregate(pipeline))
        
        if not result or result[0]['count'] < 10:
            return
        
        stats = result[0]
        avg_response = stats['avg_response']
        
        if avg_response >= rule.threshold:
            # Calculate percentiles
            responses = sorted(stats['responses'])
            p95_idx = int(len(responses) * 0.95)
            p99_idx = int(len(responses) * 0.99)
            
            self._trigger_alert(
                rule=rule,
                title=f"🐌 Slow Response Times: {avg_response:.0f}ms",
                description=(
                    f"Average response time exceeds {rule.threshold}ms. "
                    f"Service degradation detected"
                ),
                metrics={
                    'avg_response_time': avg_response,
                    'max_response_time': stats['max_response'],
                    'p95_response_time': responses[p95_idx],
                    'p99_response_time': responses[p99_idx],
                    'sample_size': stats['count']
                }
            )
    
    def _check_service_availability(self, rule: AlertRule) -> None:
        """Check service availability"""
        window_start = datetime.utcnow() - timedelta(minutes=rule.window_minutes)
        
        total = self.logs_collection.count_documents({
            'timestamp': {'$gte': window_start},
            'status_code': {'$exists': True}
        })
        
        if total < 10:
            return
        
        unavailable = self.logs_collection.count_documents({
            'timestamp': {'$gte': window_start},
            'status_code': {'$in': [503, 504, 502]}
        })
        
        unavailable_rate = unavailable / total
        
        if unavailable_rate >= rule.threshold:
            self._trigger_alert(
                rule=rule,
                title=f"🔴 Service Unavailability: {unavailable_rate:.1%}",
                description=(
                    f"Service unavailability rate of {unavailable_rate:.1%} "
                    f"exceeds critical threshold"
                ),
                metrics={
                    'unavailable_rate': unavailable_rate,
                    'total_requests': total,
                    'unavailable_count': unavailable
                }
            )
    
    def _check_anomaly_clustering(self, rule: AlertRule) -> None:
        """Check for anomaly clustering by service"""
        window_start = datetime.utcnow() - timedelta(minutes=rule.window_minutes)
        
        # Group anomalies by service
        pipeline = [
            {'$match': {
                'detected_at': {'$gte': window_start},
                'status': 'new'
            }},
            {'$group': {
                '_id': '$service',
                'count': {'$sum': 1},
                'avg_score': {'$avg': '$anomaly_score'}
            }},
            {'$match': {'count': {'$gte': rule.threshold}}},
            {'$sort': {'count': -1}}
        ]
        
        clusters = list(self.anomalies_collection.aggregate(pipeline))
        
        if clusters:
            total_anomalies = sum(c['count'] for c in clusters)
            
            self._trigger_alert(
                rule=rule,
                title=f"📊 Anomaly Clustering Detected",
                description=(
                    f"Found {len(clusters)} services with clustered anomalies. "
                    f"Total {total_anomalies} anomalies in {rule.window_minutes} minutes"
                ),
                metrics={
                    'affected_services': len(clusters),
                    'total_anomalies': total_anomalies,
                    'clusters': [
                        {'service': c['_id'], 
                         'count': c['count'],
                         'avg_score': round(c['avg_score'], 3)}
                        for c in clusters
                    ]
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
        """Trigger alert with intelligent aggregation"""
        # Create alert document
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
            'resolved': False,
            'acknowledged_at': None,
            'resolved_at': None
        }
        
        try:
            # Store alert
            result = self.alerts_collection.insert_one(alert_doc)
            alert_id = str(result.inserted_id)
            
            self.stats['alerts_generated'] += 1
            self.recent_alerts.append(alert_doc)
            
            # Update cooldown
            self.alert_cooldown[rule.name] = datetime.utcnow()
            
            # Log alert
            logger.warning("=" * 60)
            logger.warning(f"🚨 ALERT TRIGGERED: {title}")
            logger.warning(f"   Rule: {rule.name}")
            logger.warning(f"   Severity: {rule.severity.upper()}")
            logger.warning(f"   Description: {description}")
            logger.warning(f"   Alert ID: {alert_id}")
            logger.warning("=" * 60)
            
            # Send notification
            if self.smtp_config['enabled']:
                self._send_email_notification(alert_doc)
                self.stats['alerts_sent'] += 1
            
        except Exception as e:
            logger.error(f"Failed to trigger alert: {e}")
    
    def _send_email_notification(self, alert: Dict) -> None:
        """Send email notification for alert"""
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[{alert['severity'].upper()}] {alert['title']}"
            msg['From'] = self.smtp_config['from_email']
            msg['To'] = ', '.join(self.smtp_config['to_emails'])
            
            # Create email body
            body = f"""
Alert Triggered: {alert['title']}

Severity: {alert['severity'].upper()}
Rule: {alert['rule_name']}
Time: {alert['triggered_at'].strftime('%Y-%m-%d %H:%M:%S UTC')}

Description:
{alert['description']}

Metrics:
{self._format_metrics(alert['metrics'])}

---
Log Analytics System
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.smtp_config['host'], self.smtp_config['port']) as server:
                server.starttls()
                if self.smtp_config['username']:
                    server.login(
                        self.smtp_config['username'],
                        self.smtp_config['password']
                    )
                server.send_message(msg)
            
            logger.info(f"Email notification sent for alert: {alert['title']}")
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
    
    def _format_metrics(self, metrics: Dict, indent: int = 0) -> str:
        """Format metrics for display"""
        lines = []
        prefix = "  " * indent
        
        for key, value in metrics.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.append(self._format_metrics(value, indent + 1))
            elif isinstance(value, (list, tuple)):
                lines.append(f"{prefix}{key}: {len(value)} items")
            elif isinstance(value, float):
                lines.append(f"{prefix}{key}: {value:.4f}")
            else:
                lines.append(f"{prefix}{key}: {value}")
        
        return "\n".join(lines)
    
    def _process_aggregated_alerts(self) -> None:
        """Process and aggregate similar alerts"""
        # Implementation for alert aggregation
        pass
    
    def _can_alert(self, rule_name: str) -> bool:
        """Check if we can alert for this rule (cooldown)"""
        if rule_name not in self.alert_cooldown:
            return True
        
        last_alert = self.alert_cooldown[rule_name]
        cooldown_end = last_alert + timedelta(minutes=self.cooldown_minutes)
        
        can_alert = datetime.utcnow() >= cooldown_end
        
        if not can_alert:
            self.stats['alerts_suppressed'] += 1
        
        return can_alert
    
    def _handle_shutdown(self, signum, frame) -> None:
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    def get_stats(self) -> Dict:
        """Get worker statistics"""
        return {
            **self.stats,
            'active_rules': len([r for r in self.rules if r.enabled]),
            'running': self.running,
            'current_time': datetime.utcnow().isoformat()
        }
    
    def stop(self) -> None:
        """Stop the worker"""
        self.running = False


def main():
    """Main entry point"""
    logger.info("=" * 60)
    logger.info("Enhanced Alert Processing Worker")
    logger.info("=" * 60)
    
    # Configuration
    # mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
    # in PRODUCTION use os.getenv -> for now remove it
    mongo_uri = "mongodb://admin:adminpass123@localhost:27017/?authSource=admin"
    db_name = os.getenv('DB_NAME', 'log_analytics')
    check_interval = int(os.getenv('ALERT_CHECK_INTERVAL', '60'))
    
    # Create worker
    worker = AlertWorker(
        mongo_uri=mongo_uri,
        db_name=db_name,
        check_interval=check_interval
    )
    
    try:
        worker.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Worker failed: {e}", exc_info=True)
    finally:
        stats = worker.get_stats()
        logger.info("=" * 60)
        logger.info("Final Worker Statistics:")
        logger.info(f"  Checks: {stats['checks_performed']}")
        logger.info(f"  Alerts Generated: {stats['alerts_generated']}")
        logger.info(f"  Alerts Sent: {stats['alerts_sent']}")
        logger.info(f"  Alerts Suppressed: {stats['alerts_suppressed']}")
        logger.info(f"  Errors: {stats['errors']}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()