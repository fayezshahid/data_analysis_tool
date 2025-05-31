import pytest
import pandas as pd
from core.analyzer import DataAnalyzer

class MockDataManager:
    def __init__(self):
        self.datasets = {}
        self.metadata = {}
        self.undo_stacks = {}
    
    def _get_dataset(self, name):
        return self.datasets.get(name)
    
    def _dataset_exists(self, name):
        return name in self.datasets
    
    def _save_metadata(self):
        # dummy save metadata method
        pass

@pytest.fixture
def sample_data_manager():
    dm = MockDataManager()
    
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 4],
        'B': ['x', 'y', 'x', 'z', 'x'],
        'C': [10.0, None, 30.0, 40.0, 40.0],
        'D': [None, None, None, None, None]
    })
    
    dm.datasets['test'] = df.copy()
    dm.metadata['test'] = {'rows': len(df), 'file_path': None}
    dm.undo_stacks['test'] = []
    return dm

def test_summary_statistics(sample_data_manager):
    analyzer = DataAnalyzer(sample_data_manager)
    success = analyzer.summary_statistics('test')
    assert success is True
    
    # Test with non-existing dataset
    assert analyzer.summary_statistics('nonexistent') is False

def test_missing_data_report(sample_data_manager):
    analyzer = DataAnalyzer(sample_data_manager)
    success = analyzer.missing_data_report('test')
    assert success is True
    
    # Test with non-existing dataset
    assert analyzer.missing_data_report('nonexistent') is False

def test_frequency_counts_all(sample_data_manager):
    analyzer = DataAnalyzer(sample_data_manager)
    success = analyzer.frequency_counts('test')
    assert success is True

def test_frequency_counts_column(sample_data_manager):
    analyzer = DataAnalyzer(sample_data_manager)
    # Valid column
    assert analyzer.frequency_counts('test', column='B') is True
    # Invalid column
    assert analyzer.frequency_counts('test', column='InvalidCol') is False

def test_remove_duplicates(sample_data_manager):
    analyzer = DataAnalyzer(sample_data_manager)
    success = analyzer.remove_duplicates('test')
    assert success is True
    # Check duplicates removed
    df = sample_data_manager.datasets['test']
    assert df.duplicated().sum() == 0

def test_handle_missing_values_remove(sample_data_manager):
    analyzer = DataAnalyzer(sample_data_manager)
    success = analyzer.handle_missing_values('test', method='remove')
    assert success is True
    df = sample_data_manager.datasets['test']
    assert df.isnull().sum().sum() == 0

def test_handle_missing_values_fill_mean(sample_data_manager):
    analyzer = DataAnalyzer(sample_data_manager)
    success = analyzer.handle_missing_values('test', method='fill_mean')
    assert success is True
    df = sample_data_manager.datasets['test']
    assert df['C'].isnull().sum() == 0

def test_handle_missing_values_fill_mode(sample_data_manager):
    analyzer = DataAnalyzer(sample_data_manager)
    success = analyzer.handle_missing_values('test', method='fill_mode')
    assert success is True
    df = sample_data_manager.datasets['test']
    assert df['B'].isnull().sum() == 0

def test_filter_data_and_display(sample_data_manager):
    analyzer = DataAnalyzer(sample_data_manager)
    success = analyzer.filter_data_and_display('test', column='A', condition='>', value=2)
    assert success is True
    # Invalid column
    assert analyzer.filter_data_and_display('test', column='Invalid', condition='>', value=2) is False

def test_filter_data_and_modify(sample_data_manager):
    analyzer = DataAnalyzer(sample_data_manager)
    success = analyzer.filter_data_and_modify('test', column='A', condition='>', value=2)
    assert success is True
    df = sample_data_manager.datasets['test']
    assert all(df['A'] > 2)
    # Undo should revert
    undo_success = analyzer.undo_last_clean('test')
    assert undo_success is True
    df_restored = sample_data_manager.datasets['test']
    assert len(df_restored) > 0

def test_undo_no_stack(sample_data_manager):
    analyzer = DataAnalyzer(sample_data_manager)
    # Clear undo stack
    sample_data_manager.undo_stacks['test'] = []
    assert analyzer.undo_last_clean('test') is False
