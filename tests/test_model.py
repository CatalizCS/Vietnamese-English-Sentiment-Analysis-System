import pytest
import numpy as np
from src.models.model_trainer import EnhancedModelTrainer
from src.config import Config

@pytest.fixture
def trainer():
    config = Config()
    return EnhancedModelTrainer('vi', config)

def test_model_creation():
    config = Config()
    trainer = EnhancedModelTrainer('vi', config)
    model = trainer.create_ensemble_model()
    assert model is not None

def test_model_training():
    config = Config()
    trainer = EnhancedModelTrainer('vi', config)
    
    # Create dummy data
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    
    model = trainer.train_with_grid_search(X, y)
    assert model is not None