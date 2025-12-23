"""
Test cases for Log API endpoints
"""
import pytest
from datetime import datetime, timedelta
from fastapi import status


class TestLogIngestion:
    """Test log ingestion endpoints"""
    
    @pytest.mark.asyncio
    async def test_ingest_single_log_success(self, async_client, sample_log_entry):
        """Test successful single log ingestion"""
        response = await async_client.post(
            "/api/v1/logs/ingest",
            json=sample_log_entry
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["status"] == "success"
        assert "log_id" in data
        assert data["message"] == "Log ingested successfully"
    
    @pytest.mark.asyncio
    async def test_ingest_log_missing_required_field(self, async_client):
        """Test ingestion with missing required field"""
        incomplete_log = {
            "level": "ERROR",
            "message": "Test message"
            # Missing: timestamp, service
        }
        
        response = await async_client.post(
            "/api/v1/logs/ingest",
            json=incomplete_log
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    @pytest.mark.asyncio
    async def test_ingest_log_invalid_level(self, async_client, sample_log_entry):
        """Test ingestion with invalid log level"""
        sample_log_entry["level"] = "INVALID_LEVEL"
        
        response = await async_client.post(
            "/api/v1/logs/ingest",
            json=sample_log_entry
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    @pytest.mark.asyncio
    async def test_bulk_ingest_success(self, async_client, sample_log_batch):
        """Test successful bulk log ingestion"""
        logs = sample_log_batch(count=10)
        
        response = await async_client.post(
            "/api/v1/logs/bulk",
            json={"logs": logs}
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["status"] == "success"
        assert data["ingested_count"] == 10
        assert data["failed_count"] == 0
        assert len(data["log_ids"]) == 10
    
    @pytest.mark.asyncio
    async def test_bulk_ingest_partial_failure(self, async_client, sample_log_batch):
        """Test bulk ingestion with some invalid logs"""
        logs = sample_log_batch(count=5)
        
        # Add invalid log
        logs.append({"invalid": "log"})
        
        response = await async_client.post(
            "/api/v1/logs/bulk",
            json={"logs": logs}
        )
        
        # Should accept valid logs and report failures
        data = response.json()
        assert data["ingested_count"] == 5
        assert data["failed_count"] == 1
    
    @pytest.mark.asyncio
    async def test_ingest_with_metadata(self, async_client):
        """Test log ingestion with custom metadata"""
        log_with_metadata = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": "INFO",
            "service": "test-service",
            "message": "Test message",
            "metadata": {
                "user_id": "user_123",
                "session_id": "sess_456",
                "custom_field": "custom_value"
            }
        }
        
        response = await async_client.post(
            "/api/v1/logs/ingest",
            json=log_with_metadata
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["status"] == "success"


class TestLogRetrieval:
    """Test log retrieval endpoints"""
    
    @pytest.mark.asyncio
    async def test_get_logs_default(self, async_client, test_mongodb, create_test_logs):
        """Test getting logs with default parameters"""
        # Create test logs
        await create_test_logs(test_mongodb, count=50)
        
        response = await async_client.get("/api/v1/logs")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "logs" in data
        assert len(data["logs"]) <= 100  # Default limit
        assert data["total"] == 50
    
    @pytest.mark.asyncio
    async def test_get_logs_with_filters(self, async_client, test_mongodb, create_test_logs):
        """Test getting logs with filters"""
        # Create logs with specific level
        await create_test_logs(test_mongodb, count=10, level="ERROR")
        await create_test_logs(test_mongodb, count=20, level="INFO")
        
        response = await async_client.get(
            "/api/v1/logs",
            params={"level": "ERROR"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["logs"]) == 10
        assert all(log["level"] == "ERROR" for log in data["logs"])
    
    @pytest.mark.asyncio
    async def test_get_logs_with_service_filter(self, async_client, test_mongodb, create_test_logs):
        """Test filtering by service"""
        await create_test_logs(test_mongodb, count=5, service="service-a")
        await create_test_logs(test_mongodb, count=10, service="service-b")
        
        response = await async_client.get(
            "/api/v1/logs",
            params={"service": "service-b"}
        )
        
        data = response.json()
        assert len(data["logs"]) == 10
        assert all(log["service"] == "service-b" for log in data["logs"])
    
    @pytest.mark.asyncio
    async def test_get_logs_with_date_range(self, async_client, test_mongodb):
        """Test filtering by date range"""
        # Create logs in different time periods
        now = datetime.utcnow()
        yesterday = now - timedelta(days=1)
        
        logs_collection = test_mongodb['logs']
        await logs_collection.insert_many([
            {
                "timestamp": yesterday,
                "level": "INFO",
                "service": "test",
                "message": "Old log"
            },
            {
                "timestamp": now,
                "level": "INFO",
                "service": "test",
                "message": "New log"
            }
        ])
        
        response = await async_client.get(
            "/api/v1/logs",
            params={
                "start_date": (now - timedelta(hours=1)).isoformat(),
                "end_date": now.isoformat()
            }
        )
        
        data = response.json()
        assert len(data["logs"]) == 1
        assert data["logs"][0]["message"] == "New log"
    
    @pytest.mark.asyncio
    async def test_get_logs_pagination(self, async_client, test_mongodb, create_test_logs):
        """Test pagination"""
        await create_test_logs(test_mongodb, count=150)
        
        # First page
        response1 = await async_client.get(
            "/api/v1/logs",
            params={"limit": 50, "skip": 0}
        )
        data1 = response1.json()
        assert len(data1["logs"]) == 50
        assert data1["has_more"] is True
        
        # Second page
        response2 = await async_client.get(
            "/api/v1/logs",
            params={"limit": 50, "skip": 50}
        )
        data2 = response2.json()
        assert len(data2["logs"]) == 50
        
        # Verify different logs
        log_ids_1 = {log["_id"] for log in data1["logs"]}
        log_ids_2 = {log["_id"] for log in data2["logs"]}
        assert log_ids_1.isdisjoint(log_ids_2)
    
    @pytest.mark.asyncio
    async def test_get_log_by_id(self, async_client, test_mongodb):
        """Test getting specific log by ID"""
        # Insert a log
        logs_collection = test_mongodb['logs']
        result = await logs_collection.insert_one({
            "timestamp": datetime.utcnow(),
            "level": "INFO",
            "service": "test-service",
            "message": "Test message"
        })
        log_id = str(result.inserted_id)
        
        response = await async_client.get(f"/api/v1/logs/{log_id}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["_id"] == log_id
        assert data["message"] == "Test message"
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_log(self, async_client):
        """Test getting log that doesn't exist"""
        fake_id = "507f1f77bcf86cd799439011"
        
        response = await async_client.get(f"/api/v1/logs/{fake_id}")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestLogSearch:
    """Test log search functionality"""
    
    @pytest.mark.asyncio
    async def test_search_logs_by_text(self, async_client, test_mongodb):
        """Test full-text search"""
        logs_collection = test_mongodb['logs']
        await logs_collection.insert_many([
            {
                "timestamp": datetime.utcnow(),
                "level": "ERROR",
                "service": "payment",
                "message": "Payment processing failed"
            },
            {
                "timestamp": datetime.utcnow(),
                "level": "INFO",
                "service": "payment",
                "message": "Payment successful"
            },
            {
                "timestamp": datetime.utcnow(),
                "level": "INFO",
                "service": "user",
                "message": "User login successful"
            }
        ])
        
        response = await async_client.post(
            "/api/v1/logs/search",
            json={"query": "payment failed"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["results"]) >= 1
        assert "payment" in data["results"][0]["message"].lower()
    
    @pytest.mark.asyncio
    async def test_search_with_filters(self, async_client, test_mongodb):
        """Test search with additional filters"""
        logs_collection = test_mongodb['logs']
        await logs_collection.insert_many([
            {
                "timestamp": datetime.utcnow(),
                "level": "ERROR",
                "service": "payment",
                "message": "Payment failed",
                "status_code": 500
            },
            {
                "timestamp": datetime.utcnow(),
                "level": "ERROR",
                "service": "user",
                "message": "Authentication failed",
                "status_code": 401
            }
        ])
        
        response = await async_client.post(
            "/api/v1/logs/search",
            json={
                "query": "failed",
                "filters": {
                    "level": ["ERROR"],
                    "service": ["payment"]
                }
            }
        )
        
        data = response.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["service"] == "payment"
    
    @pytest.mark.asyncio
    async def test_search_empty_query(self, async_client):
        """Test search with empty query"""
        response = await async_client.post(
            "/api/v1/logs/search",
            json={"query": ""}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestLogStatistics:
    """Test log statistics endpoints"""
    
    @pytest.mark.asyncio
    async def test_get_log_stats(self, async_client, test_mongodb, create_test_logs):
        """Test getting log statistics"""
        await create_test_logs(test_mongodb, count=50, level="INFO")
        await create_test_logs(test_mongodb, count=30, level="WARN")
        await create_test_logs(test_mongodb, count=20, level="ERROR")
        
        response = await async_client.get("/api/v1/logs/stats")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total_logs"] == 100
        assert data["logs_by_level"]["INFO"] == 50
        assert data["logs_by_level"]["WARN"] == 30
        assert data["logs_by_level"]["ERROR"] == 20


class TestPerformance:
    """Performance and load tests"""
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_bulk_ingest_performance(self, async_client, sample_log_batch, benchmark_timer):
        """Test performance of bulk ingestion"""
        logs = sample_log_batch(count=1000)
        
        benchmark_timer.start()
        response = await async_client.post(
            "/api/v1/logs/bulk",
            json={"logs": logs}
        )
        elapsed = benchmark_timer.stop()
        
        assert response.status_code == status.HTTP_201_CREATED
        assert elapsed < 5.0  # Should complete in under 5 seconds
        
        # Calculate throughput
        throughput = 1000 / elapsed
        print(f"\nThroughput: {throughput:.2f} logs/second")
        assert throughput > 200  # At least 200 logs/second
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_query_performance(self, async_client, test_mongodb, create_test_logs, benchmark_timer):
        """Test query performance with large dataset"""
        # Create large dataset
        await create_test_logs(test_mongodb, count=10000)
        
        benchmark_timer.start()
        response = await async_client.get(
            "/api/v1/logs",
            params={"level": "ERROR", "limit": 100}
        )
        elapsed = benchmark_timer.stop()
        
        assert response.status_code == status.HTTP_200_OK
        assert elapsed < 1.0  # Should complete in under 1 second
        print(f"\nQuery time: {elapsed:.3f}s")


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    @pytest.mark.asyncio
    async def test_ingest_very_long_message(self, async_client, sample_log_entry):
        """Test ingesting log with very long message"""
        sample_log_entry["message"] = "A" * 10000  # 10KB message
        
        response = await async_client.post(
            "/api/v1/logs/ingest",
            json=sample_log_entry
        )
        
        # Should accept but might truncate
        assert response.status_code in [status.HTTP_201_CREATED, status.HTTP_400_BAD_REQUEST]
    
    @pytest.mark.asyncio
    async def test_ingest_future_timestamp(self, async_client, sample_log_entry):
        """Test ingesting log with future timestamp"""
        future_time = datetime.utcnow() + timedelta(days=365)
        sample_log_entry["timestamp"] = future_time.isoformat()
        
        response = await async_client.post(
            "/api/v1/logs/ingest",
            json=sample_log_entry
        )
        
        # Should handle gracefully (accept or reject with clear error)
        assert response.status_code in [status.HTTP_201_CREATED, status.HTTP_400_BAD_REQUEST]
    
    @pytest.mark.asyncio
    async def test_concurrent_ingestion(self, async_client, sample_log_batch):
        """Test concurrent log ingestion"""
        import asyncio
        
        logs = sample_log_batch(count=10)
        
        # Send multiple requests concurrently
        tasks = [
            async_client.post("/api/v1/logs/ingest", json=log)
            for log in logs
        ]
        
        responses = await asyncio.gather(*tasks)
        
        # All should succeed
        assert all(r.status_code == status.HTTP_201_CREATED for r in responses)
    
    @pytest.mark.asyncio
    async def test_special_characters_in_message(self, async_client, sample_log_entry):
        """Test handling of special characters"""
        sample_log_entry["message"] = "Test with special chars: 你好 🚀 <script>alert('xss')</script>"
        
        response = await async_client.post(
            "/api/v1/logs/ingest",
            json=sample_log_entry
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        
        # Verify stored correctly
        log_id = response.json()["log_id"]
        get_response = await async_client.get(f"/api/v1/logs/{log_id}")
        stored_message = get_response.json()["message"]
        
        # Should be stored safely (escaped or sanitized)
        assert "script" not in stored_message or "<script>" not in stored_message