// MongoDB Initialization Script
// This script runs when MongoDB container starts for the first time

print('🔧 Starting MongoDB initialization...');

// Switch to log_analytics database
db = db.getSiblingDB('log_analytics');

print('✓ Switched to log_analytics database');

// Create collections with validators
print('Creating collections with schema validation...');

// Logs Collection
db.createCollection('logs', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['timestamp', 'level', 'service', 'message', 'created_at'],
      properties: {
        timestamp: {
          bsonType: 'date',
          description: 'must be a date and is required'
        },
        level: {
          enum: ['DEBUG', 'INFO', 'WARN', 'WARNING', 'ERROR', 'CRITICAL'],
          description: 'must be a valid log level and is required'
        },
        service: {
          bsonType: 'string',
          description: 'must be a string and is required'
        },
        message: {
          bsonType: 'string',
          description: 'must be a string and is required'
        },
        endpoint: {
          bsonType: 'string',
          description: 'API endpoint if applicable'
        },
        response_time: {
          bsonType: 'double',
          description: 'response time in milliseconds'
        },
        status_code: {
          bsonType: 'int',
          description: 'HTTP status code'
        },
        user_id: {
          bsonType: 'string',
          description: 'user identifier'
        },
        ip_address: {
          bsonType: 'string',
          description: 'client IP address'
        },
        created_at: {
          bsonType: 'date',
          description: 'creation timestamp'
        },
        processed: {
          bsonType: 'bool',
          description: 'processing status'
        },
        anomaly_checked: {
          bsonType: 'bool',
          description: 'anomaly detection status'
        }
      }
    }
  }
});
print('✓ Created logs collection');

// Anomalies Collection
db.createCollection('anomalies', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['log_id', 'timestamp', 'anomaly_score', 'detected_at'],
      properties: {
        log_id: {
          bsonType: 'objectId',
          description: 'reference to log entry'
        },
        timestamp: {
          bsonType: 'date',
          description: 'timestamp of anomalous event'
        },
        anomaly_score: {
          bsonType: 'double',
          minimum: 0,
          maximum: 1,
          description: 'anomaly score between 0 and 1'
        },
        anomaly_type: {
          bsonType: 'string',
          description: 'type of anomaly detected'
        },
        model_version: {
          bsonType: 'string',
          description: 'ML model version used'
        },
        features: {
          bsonType: 'object',
          description: 'extracted features'
        },
        detected_at: {
          bsonType: 'date',
          description: 'when anomaly was detected'
        },
        is_confirmed: {
          bsonType: 'bool',
          description: 'whether anomaly is confirmed'
        }
      }
    }
  }
});
print('✓ Created anomalies collection');

// Alerts Collection
db.createCollection('alerts', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['alert_type', 'severity', 'message', 'created_at', 'status'],
      properties: {
        alert_type: {
          enum: ['anomaly', 'threshold', 'pattern', 'system'],
          description: 'type of alert'
        },
        severity: {
          enum: ['low', 'medium', 'high', 'critical'],
          description: 'alert severity level'
        },
        message: {
          bsonType: 'string',
          description: 'alert message'
        },
        details: {
          bsonType: 'object',
          description: 'additional alert details'
        },
        related_log_ids: {
          bsonType: 'array',
          description: 'array of related log IDs'
        },
        created_at: {
          bsonType: 'date',
          description: 'when alert was created'
        },
        status: {
          enum: ['pending', 'sent', 'acknowledged', 'resolved', 'ignored'],
          description: 'alert status'
        },
        notified_at: {
          bsonType: 'date',
          description: 'when notification was sent'
        },
        acknowledged_at: {
          bsonType: 'date',
          description: 'when alert was acknowledged'
        },
        resolved_at: {
          bsonType: 'date',
          description: 'when alert was resolved'
        }
      }
    }
  }
});
print('✓ Created alerts collection');

// Models Collection (for ML model metadata)
db.createCollection('models', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['model_name', 'version', 'created_at', 'is_active'],
      properties: {
        model_name: {
          bsonType: 'string',
          description: 'name of the ML model'
        },
        version: {
          bsonType: 'string',
          description: 'model version'
        },
        model_type: {
          bsonType: 'string',
          description: 'type of model (e.g., isolation_forest, lstm)'
        },
        hyperparameters: {
          bsonType: 'object',
          description: 'model hyperparameters'
        },
        metrics: {
          bsonType: 'object',
          description: 'model performance metrics'
        },
        created_at: {
          bsonType: 'date',
          description: 'when model was created'
        },
        is_active: {
          bsonType: 'bool',
          description: 'whether model is currently active'
        },
        file_path: {
          bsonType: 'string',
          description: 'path to saved model file'
        }
      }
    }
  }
});
print('✓ Created models collection');

// Create Indexes
print('Creating indexes for optimal performance...');

// Logs indexes
db.logs.createIndex({ 'timestamp': -1 });
db.logs.createIndex({ 'level': 1 });
db.logs.createIndex({ 'service': 1 });
db.logs.createIndex({ 'service': 1, 'timestamp': -1 });
db.logs.createIndex({ 'level': 1, 'timestamp': -1 });
db.logs.createIndex({ 'created_at': -1 });
db.logs.createIndex({ 'processed': 1 });
db.logs.createIndex({ 'anomaly_checked': 1 });
db.logs.createIndex({ 'user_id': 1 });
db.logs.createIndex({ 'ip_address': 1 });
// Text index for full-text search on message
db.logs.createIndex({ 'message': 'text' });
print('✓ Created indexes for logs collection');

// Anomalies indexes
db.anomalies.createIndex({ 'timestamp': -1 });
db.anomalies.createIndex({ 'anomaly_score': -1 });
db.anomalies.createIndex({ 'log_id': 1 });
db.anomalies.createIndex({ 'anomaly_type': 1 });
db.anomalies.createIndex({ 'detected_at': -1 });
db.anomalies.createIndex({ 'is_confirmed': 1 });
print('✓ Created indexes for anomalies collection');

// Alerts indexes
db.alerts.createIndex({ 'created_at': -1 });
db.alerts.createIndex({ 'status': 1 });
db.alerts.createIndex({ 'severity': 1 });
db.alerts.createIndex({ 'alert_type': 1 });
db.alerts.createIndex({ 'status': 1, 'created_at': -1 });
print('✓ Created indexes for alerts collection');

// Models indexes
db.models.createIndex({ 'model_name': 1, 'version': 1 }, { unique: true });
db.models.createIndex({ 'is_active': 1 });
db.models.createIndex({ 'created_at': -1 });
print('✓ Created indexes for models collection');

// Create a default admin user (optional)
// db.users.insertOne({
//   username: 'admin',
//   email: 'admin@loganalytics.com',
//   role: 'admin',
//   created_at: new Date()
// });

// Insert sample model metadata
db.models.insertOne({
  model_name: 'isolation_forest',
  version: 'v1.0.0',
  model_type: 'isolation_forest',
  hyperparameters: {
    n_estimators: 100,
    contamination: 0.1,
    max_samples: 'auto'
  },
  metrics: {
    precision: 0.0,
    recall: 0.0,
    f1_score: 0.0
  },
  created_at: new Date(),
  is_active: true,
  file_path: '/app/saved_models/isolation_forest_v1.pkl'
});
print('✓ Inserted default model metadata');

// Print database statistics
print('\n📊 Database Statistics:');
print('Collections created: ' + db.getCollectionNames().length);
print('Logs indexes: ' + db.logs.getIndexes().length);
print('Anomalies indexes: ' + db.anomalies.getIndexes().length);
print('Alerts indexes: ' + db.alerts.getIndexes().length);
print('Models indexes: ' + db.models.getIndexes().length);

print('\n✅ MongoDB initialization complete!\n');