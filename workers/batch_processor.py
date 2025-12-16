"""
Batch Processing Worker - Process historical logs in batches
"""
import sys
import os
import time
import signal
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from pymongo import MongoClient
from ml.inference.anomaly_detector import AnomalyDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [BATCH_PROCESSOR] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchProcessor:
    """Process historical logs in batches for anomaly detection"""
    
    def __init__(
        self,
        mongo_uri: str = "mongodb://localhost:27017",
        db_name: str = "log_analytics",
        model_path: Optional[str] = None,
        batch_size: int = 100
    ):
        """
        Initialize batch processor
        
        Args:
            mongo_uri: MongoDB connection URI
            db_name: Database name
            model_path: Path to trained model
            batch_size: Number of logs to process per batch
        """
        self.mongo_client = MongoClient(mongo_uri)
        self.db = self.mongo_client[db_name]
        
        self.logs_collection = self.db['logs']
        self.anomalies_collection = self.db['anomalies']
        self.batch_jobs_collection = self.db['batch_jobs']
        
        self.batch_size = batch_size
        
        # Initialize detector
        self.detector = AnomalyDetector(model_path=model_path)
        
        self.running = False
        self.stats = {
            'jobs_completed': 0,
            'logs_processed': 0,
            'anomalies_detected': 0,
            'errors': 0,
            'started_at': None
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
    
    def create_batch_job(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        service: Optional[str] = None,
        reprocess: bool = False
    ) -> str:
        """
        Create a batch processing job
        
        Args:
            start_time: Start timestamp for log range
            end_time: End timestamp for log range
            service: Specific service to process
            reprocess: Whether to reprocess already analyzed logs
            
        Returns:
            Job ID
        """
        job = {
            'status': 'pending',
            'created_at': datetime.utcnow(),
            'started_at': None,
            'completed_at': None,
            'parameters': {
                'start_time': start_time,
                'end_time': end_time,
                'service': service,
                'reprocess': reprocess
            },
            'stats': {
                'logs_processed': 0,
                'anomalies_detected': 0,
                'errors': 0
            }
        }
        
        result = self.batch_jobs_collection.insert_one(job)
        job_id = str(result.inserted_id)
        
        logger.info(f"Created batch job: {job_id}")
        return job_id
    
    def process_job(self, job_id: str) -> Dict:
        """
        Process a specific batch job
        
        Args:
            job_id: Job ID to process
            
        Returns:
            Job results
        """
        from bson import ObjectId
        
        # Get job
        job = self.batch_jobs_collection.find_one({'_id': ObjectId(job_id)})
        if not job:
            raise ValueError(f"Job not found: {job_id}")
        
        if job['status'] == 'running':
            raise ValueError(f"Job already running: {job_id}")
        
        # Update status
        self.batch_jobs_collection.update_one(
            {'_id': ObjectId(job_id)},
            {
                '$set': {
                    'status': 'running',
                    'started_at': datetime.utcnow()
                }
            }
        )
        
        try:
            # Build query
            query = self._build_query(job['parameters'])
            
            # Get total count
            total_logs = self.logs_collection.count_documents(query)
            logger.info(f"Processing {total_logs} logs for job {job_id}")
            
            # Process in batches
            processed = 0
            anomalies_found = 0
            
            cursor = self.logs_collection.find(query).batch_size(self.batch_size)
            
            batch = []
            for log in cursor:
                batch.append(log)
                
                if len(batch) >= self.batch_size:
                    anomalies = self._process_batch(batch)
                    processed += len(batch)
                    anomalies_found += anomalies
                    
                    # Update progress
                    self._update_job_progress(job_id, processed, anomalies_found)
                    
                    batch = []
                    
                    # Log progress
                    if processed % 1000 == 0:
                        logger.info(f"Job {job_id}: {processed}/{total_logs} logs processed")
            
            # Process remaining
            if batch:
                anomalies = self._process_batch(batch)
                processed += len(batch)
                anomalies_found += anomalies
            
            # Mark complete
            self.batch_jobs_collection.update_one(
                {'_id': ObjectId(job_id)},
                {
                    '$set': {
                        'status': 'completed',
                        'completed_at': datetime.utcnow(),
                        'stats': {
                            'logs_processed': processed,
                            'anomalies_detected': anomalies_found,
                            'errors': 0
                        }
                    }
                }
            )
            
            logger.info(
                f"Job {job_id} completed: {processed} logs, "
                f"{anomalies_found} anomalies"
            )
            
            return {
                'job_id': job_id,
                'status': 'completed',
                'logs_processed': processed,
                'anomalies_detected': anomalies_found
            }
            
        except Exception as e:
            logger.error(f"Job {job_id} failed: {str(e)}", exc_info=True)
            
            # Mark failed
            self.batch_jobs_collection.update_one(
                {'_id': ObjectId(job_id)},
                {
                    '$set': {
                        'status': 'failed',
                        'error': str(e),
                        'completed_at': datetime.utcnow()
                    }
                }
            )
            
            raise
    
    def _build_query(self, parameters: Dict) -> Dict:
        """Build MongoDB query from job parameters"""
        query = {}
        
        # Time range
        if parameters.get('start_time') or parameters.get('end_time'):
            query['timestamp'] = {}
            if parameters.get('start_time'):
                query['timestamp']['$gte'] = parameters['start_time']
            if parameters.get('end_time'):
                query['timestamp']['$lte'] = parameters['end_time']
        
        # Service filter
        if parameters.get('service'):
            query['service'] = parameters['service']
        
        # Reprocess filter
        if not parameters.get('reprocess', False):
            # Only process logs not already analyzed
            query['analyzed'] = {'$ne': True}
        
        return query
    
    def _process_batch(self, logs: List[Dict]) -> int:
        """
        Process a batch of logs
        
        Args:
            logs: List of log documents
            
        Returns:
            Number of anomalies detected
        """
        anomalies_detected = 0
        
        for log in logs:
            try:
                # Convert ObjectId to string
                log['_id'] = str(log['_id'])
                
                # Detect anomaly
                result = self.detector.detect_anomaly(log, use_sequence=False)
                
                # Store if anomaly
                if result.get('is_anomaly', False):
                    self._store_anomaly(log, result)
                    anomalies_detected += 1
                
                # Mark as analyzed
                from bson import ObjectId
                self.logs_collection.update_one(
                    {'_id': ObjectId(log['_id'])},
                    {'$set': {'analyzed': True}}
                )
                
            except Exception as e:
                logger.error(f"Error processing log: {str(e)}")
                self.stats['errors'] += 1
        
        return anomalies_detected
    
    def _store_anomaly(self, log: Dict, detection_result: Dict) -> None:
        """Store detected anomaly"""
        anomaly_doc = {
            'log_id': log.get('_id'),
            'timestamp': log.get('timestamp', datetime.utcnow()),
            'service': log.get('service'),
            'severity': log.get('severity'),
            'message': log.get('message'),
            'status_code': log.get('status_code'),
            'response_time': log.get('response_time'),
            'anomaly_score': detection_result['anomaly_score'],
            'anomaly_severity': detection_result['severity'],
            'confidence': detection_result.get('confidence', 0),
            'detection_method': 'batch_processing',
            'detected_at': datetime.utcnow(),
            'status': 'new',
            'investigated': False,
            'false_positive': False,
            'original_log': log
        }
        
        try:
            self.anomalies_collection.insert_one(anomaly_doc)
        except Exception as e:
            logger.error(f"Failed to store anomaly: {str(e)}")
    
    def _update_job_progress(
        self,
        job_id: str,
        processed: int,
        anomalies: int
    ) -> None:
        """Update job progress"""
        from bson import ObjectId
        
        self.batch_jobs_collection.update_one(
            {'_id': ObjectId(job_id)},
            {
                '$set': {
                    'stats.logs_processed': processed,
                    'stats.anomalies_detected': anomalies
                }
            }
        )
    
    def process_pending_jobs(self) -> None:
        """Process all pending jobs"""
        logger.info("Processing pending batch jobs")
        self.running = True
        self.stats['started_at'] = datetime.utcnow().isoformat()
        
        while self.running:
            # Find pending job
            job = self.batch_jobs_collection.find_one({'status': 'pending'})
            
            if not job:
                logger.info("No pending jobs found")
                time.sleep(10)
                continue
            
            job_id = str(job['_id'])
            
            try:
                # Process job
                result = self.process_job(job_id)
                
                self.stats['jobs_completed'] += 1
                self.stats['logs_processed'] += result['logs_processed']
                self.stats['anomalies_detected'] += result['anomalies_detected']
                
            except Exception as e:
                logger.error(f"Job processing failed: {str(e)}")
                self.stats['errors'] += 1
        
        logger.info("Batch processor stopped")
    
    def get_job_status(self, job_id: str) -> Dict:
        """Get status of a batch job"""
        from bson import ObjectId
        
        job = self.batch_jobs_collection.find_one({'_id': ObjectId(job_id)})
        
        if not job:
            return {'error': 'Job not found'}
        
        job['_id'] = str(job['_id'])
        return job
    
    def list_jobs(
        self,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """List batch jobs"""
        query = {}
        if status:
            query['status'] = status
        
        jobs = list(
            self.batch_jobs_collection
            .find(query)
            .sort('created_at', -1)
            .limit(limit)
        )
        
        for job in jobs:
            job['_id'] = str(job['_id'])
        
        return jobs
    
    def _handle_shutdown(self, signum, frame) -> None:
        """Handle shutdown signals"""
        logger.info(f"Received shutdown signal: {signum}")
        self.running = False
    
    def get_stats(self) -> Dict:
        """Get processor statistics"""
        return {
            **self.stats,
            'running': self.running,
            'current_time': datetime.utcnow().isoformat()
        }


def main():
    """Main entry point"""
    logger.info("Starting Batch Processor")
    
    # Configuration
    mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
    db_name = os.getenv('DB_NAME', 'log_analytics')
    
    # Find model
    models_dir = Path(__file__).parent.parent / 'ml' / 'saved_models'
    model_files = list(models_dir.glob('isolation_forest_*.pkl'))
    
    model_path = None
    if model_files:
        model_path = str(sorted(model_files)[-1])
        logger.info(f"Using model: {model_path}")
    
    # Create processor
    processor = BatchProcessor(
        mongo_uri=mongo_uri,
        db_name=db_name,
        model_path=model_path
    )
    
    try:
        # Process pending jobs
        processor.process_pending_jobs()
    except Exception as e:
        logger.error(f"Processor failed: {str(e)}", exc_info=True)
    finally:
        stats = processor.get_stats()
        logger.info("=" * 50)
        logger.info("Batch Processor Statistics:")
        logger.info(f"  Jobs: {stats['jobs_completed']}")
        logger.info(f"  Logs: {stats['logs_processed']}")
        logger.info(f"  Anomalies: {stats['anomalies_detected']}")
        logger.info(f"  Errors: {stats['errors']}")
        logger.info("=" * 50)


if __name__ == "__main__":
    main()