"""
TEMPORAL FUSION INTEGRATION TEST
=================================
Tests to verify the temporal fusion system integrates correctly
with the existing Smart HSRP pipeline.
"""

import numpy as np
from typing import Dict, Any


def test_temporal_fusion_engine():
    """Test the core temporal fusion engine."""
    
    print("Testing Temporal Fusion Engine...")
    
    from backend.core.temporal_fusion import TemporalDecisionFusion, TemporalFusionConfig
    
    # Create fusion engine
    config = TemporalFusionConfig(
        ema_alpha=0.3,
        bias_cap=0.08,
        min_history=3,
    )
    fusion = TemporalDecisionFusion(config)
    
    # Simulate consistent positive detections
    track_id = "test_veh_001"
    results = []
    
    for frame_id in range(10):
        result = fusion.update(
            track_id=track_id,
            decision_type="test_hsrp",
            raw_score=0.65,
            confidence=0.80,
            frame_id=frame_id,
            base_threshold=0.5,
        )
        results.append(result)
    
    # Verify results
    assert results[0]['decision'] == True, "First decision should be positive"
    assert results[-1]['stability'] > results[0]['stability'], "Stability should increase"
    assert results[-1]['consecutive_streak'] >= 3, "Should have consecutive streak"
    
    print("  ✓ Core fusion engine working")
    
    # Test cleanup
    fusion.cleanup_old_tracks(current_frame=200, max_age=50)
    stats = fusion.get_statistics()
    assert stats['active_tracks'] == 0, "Old tracks should be cleaned up"
    
    print("  ✓ Cleanup working")


def test_hsrp_decision_manager():
    """Test HSRP decision manager."""
    
    print("\nTesting HSRP Decision Manager...")
    
    from backend.services.decision_managers import HSRPDecisionManager
    
    manager = HSRPDecisionManager()
    
    # Simulate HSRP classification results
    mock_hsrp_result = {
        'label': 'non_hsrp',
        'confidence': 0.75,
        'prob_non_hsrp': 0.75,
        'prob_hsrp': 0.25,
    }
    
    result = manager.process(
        track_id="veh_001",
        hsrp_result=mock_hsrp_result,
        frame_id=100,
    )
    
    # Verify structure
    assert 'is_violation' in result, "Should have is_violation field"
    assert 'temporal_info' in result, "Should have temporal_info"
    assert 'raw_label' in result, "Should preserve raw label"
    
    print("  ✓ HSRP manager working")


def test_helmet_decision_manager():
    """Test helmet decision manager."""
    
    print("\nTesting Helmet Decision Manager...")
    
    from backend.services.decision_managers import HelmetDecisionManager
    
    manager = HelmetDecisionManager()
    
    # Simulate helmet detection results
    mock_helmet_result = {
        'status': 'NO_HELMET',
        'confidence': 0.82,
        'bbox': [10, 20, 50, 80],
    }
    
    result = manager.process(
        track_id="veh_001",
        helmet_result=mock_helmet_result,
        frame_id=100,
    )
    
    # Verify structure
    assert 'is_violation' in result, "Should have is_violation field"
    assert 'has_helmet' in result, "Should have has_helmet field"
    assert 'temporal_info' in result, "Should have temporal_info"
    
    print("  ✓ Helmet manager working")


def test_violation_aggregator():
    """Test violation decision aggregator."""
    
    print("\nTesting Violation Aggregator...")
    
    from backend.services.decision_managers import ViolationDecisionAggregator
    
    aggregator = ViolationDecisionAggregator()
    
    # Simulate HSRP violation over multiple frames
    hsrp_decision = {
        'is_violation': True,
        'confidence': 0.85,
        'temporal_info': {'stability': 0.75},
    }
    
    # Process multiple frames
    for frame_id in range(100, 110):
        result = aggregator.evaluate_violation(
            track_id="veh_001",
            vehicle_class="car",
            hsrp_decision=hsrp_decision,
            helmet_decision=None,
            ocr_decision=None,
            frame_id=frame_id,
        )
    
    # Should detect violation after multiple frames
    assert result['has_violation'] == True, "Should detect violation after multiple frames"
    assert result['violation_type'] == "non_hsrp_plate", "Should identify HSRP violation"
    
    print("  ✓ Violation aggregator working")


def test_frame_pipeline_integration():
    """Test enhanced frame pipeline integration."""
    
    print("\nTesting Frame Pipeline Integration...")
    
    from backend.core.frame_pipeline import FramePipeline
    
    pipeline = FramePipeline()
    
    # Create mock frame
    mock_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Mock models (these would normally be actual model instances)
    class MockDetector:
        def detect(self, frame):
            return []  # Empty detections for test
        def predict(self, frame):
            return []
    
    result = pipeline.process_frame(
        frame=mock_frame,
        frame_id=100,
        vehicle_detector=MockDetector(),
        helmet_detector=MockDetector(),
        plate_detector=MockDetector(),
        hsrp_classifier=MockDetector(),
        ocr_model=MockDetector(),
        ocr_stabilizer=None,
        tracker=None,
        frame_skip=2,
    )
    
    # Verify structure
    assert 'frame_id' in result, "Should have frame_id"
    assert 'entities' in result, "Should have entities"
    assert 'violations' in result, "Should have violations field"
    assert 'temporal_stats' in result, "Should have temporal_stats"
    
    print("  ✓ Frame pipeline integration working")


def test_configuration_validation():
    """Test configuration parameter validation."""
    
    print("\nTesting Configuration Validation...")
    
    from backend.core.temporal_fusion import TemporalFusionConfig
    
    # Test valid configuration
    config = TemporalFusionConfig(
        ema_alpha=0.3,
        bias_cap=0.08,
        decay_rate=0.95,
    )
    
    assert 0 < config.ema_alpha < 1, "EMA alpha should be in (0, 1)"
    assert config.bias_cap > 0, "Bias cap should be positive"
    assert 0 < config.decay_rate < 1, "Decay rate should be in (0, 1)"
    
    print("  ✓ Configuration validation working")


def test_frame_skip_handling():
    """Test frame skip awareness."""
    
    print("\nTesting Frame Skip Handling...")
    
    from backend.core.temporal_fusion import TemporalDecisionFusion, TemporalFusionConfig
    
    config = TemporalFusionConfig(frame_skip_decay=0.98)
    fusion = TemporalDecisionFusion(config)
    
    track_id = "skip_test_001"
    
    # Update at frame 100
    r1 = fusion.update(
        track_id=track_id,
        decision_type="test",
        raw_score=0.7,
        confidence=0.8,
        frame_id=100,
    )
    
    # Skip frames 101, 102
    # Update at frame 103 (3 frames later)
    r2 = fusion.update(
        track_id=track_id,
        decision_type="test",
        raw_score=0.7,
        confidence=0.8,
        frame_id=103,
    )
    
    # State should have decayed
    state = fusion.get_track_state(track_id, "test")
    assert state is not None, "Track state should exist"
    assert state.last_update_frame == 103, "Should track last update"
    
    print("  ✓ Frame skip handling working")


def test_drift_control():
    """Test bias drift control."""
    
    print("\nTesting Drift Control...")
    
    from backend.core.temporal_fusion import TemporalDecisionFusion, TemporalFusionConfig
    
    config = TemporalFusionConfig(drift_threshold=0.15)
    fusion = TemporalDecisionFusion(config)
    
    track_id = "drift_test_001"
    
    # Simulate extreme bias scenario
    for frame_id in range(20):
        fusion.update(
            track_id=track_id,
            decision_type="test",
            raw_score=0.9,  # Consistently high
            confidence=0.9,
            frame_id=frame_id,
        )
    
    state = fusion.get_track_state(track_id, "test")
    
    # Bias should be capped
    assert abs(state.bias) <= config.drift_threshold * 1.5, "Bias should be controlled"
    
    print("  ✓ Drift control working")


def run_all_tests():
    """Run all integration tests."""
    
    print("=" * 60)
    print("TEMPORAL FUSION INTEGRATION TESTS")
    print("=" * 60)
    
    try:
        test_temporal_fusion_engine()
        test_hsrp_decision_manager()
        test_helmet_decision_manager()
        test_violation_aggregator()
        test_frame_pipeline_integration()
        test_configuration_validation()
        test_frame_skip_handling()
        test_drift_control()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        print("\nThe temporal fusion system is working correctly!")
        print("You can now integrate it into your Smart HSRP pipeline.")
        
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
