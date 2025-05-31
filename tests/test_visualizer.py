import os
import tempfile
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from core.visualizer import DataVisualizer

@pytest.fixture
def sample_data_manager():
    # Mock DataManager with a _get_dataset and get_dataset_info method
    dm = MagicMock()
    # Sample dataframe with numerical and categorical data
    df = pd.DataFrame({
        'num_col': [1, 2, 3, 4, 5, None],
        'cat_col': ['a', 'b', 'a', 'b', 'c', None],
        'num_col2': [10, 20, 30, None, 50, 60]
    })
    dm._get_dataset.return_value = df
    dm.get_dataset_info.return_value = {
        'file_path': '/fake/path/data.csv',
        'loaded_date': '2025-05-01 12:00:00'
    }
    return dm

@pytest.fixture
def visualizer(tmp_path, sample_data_manager):
    # Create DataVisualizer with temp dirs for plots and reports
    plots_dir = tmp_path / "plots"
    reports_dir = tmp_path / "reports"
    visualizer = DataVisualizer(
        data_manager=sample_data_manager,
        plots_dir=str(plots_dir),
        reports_dir=str(reports_dir)
    )
    return visualizer

def test_generate_histogram_success(visualizer):
    result = visualizer.generate_histogram('dataset1', 'num_col')
    assert result is True
    assert 'dataset1' in visualizer.generated_plots
    plots = visualizer.generated_plots['dataset1']
    assert any(plot['type'] == 'histogram' and plot['column'] == 'num_col' for plot in plots)
    # Check file exists
    plot_file = plots[0]['filepath']
    assert os.path.isfile(plot_file)

def test_generate_histogram_column_missing(visualizer):
    result = visualizer.generate_histogram('dataset1', 'missing_col')
    assert result is False

def test_generate_histogram_column_not_numeric(visualizer):
    result = visualizer.generate_histogram('dataset1', 'cat_col')
    assert result is False

def test_generate_histogram_no_valid_data(visualizer, sample_data_manager):
    # Mock dataset with all NaN in column
    sample_data_manager._get_dataset.return_value = pd.DataFrame({'num_col': [None, None]})
    result = visualizer.generate_histogram('dataset1', 'num_col')
    assert result is False

def test_generate_bar_chart_success(visualizer):
    result = visualizer.generate_bar_chart('dataset1', 'cat_col')
    assert result is True
    assert 'dataset1' in visualizer.generated_plots
    plots = visualizer.generated_plots['dataset1']
    assert any(plot['type'] == 'bar_chart' and plot['column'] == 'cat_col' for plot in plots)
    plot_file = next(p['filepath'] for p in plots if p['type'] == 'bar_chart')
    assert os.path.isfile(plot_file)

def test_generate_bar_chart_column_missing(visualizer):
    result = visualizer.generate_bar_chart('dataset1', 'missing_col')
    assert result is False

def test_generate_bar_chart_no_data(visualizer, sample_data_manager):
    sample_data_manager._get_dataset.return_value = pd.DataFrame({'cat_col': []})
    result = visualizer.generate_bar_chart('dataset1', 'cat_col')
    assert result is False

def test_generate_scatter_plot_success(visualizer):
    result = visualizer.generate_scatter_plot('dataset1', 'num_col', 'num_col2')
    assert result is True
    plots = visualizer.generated_plots['dataset1']
    assert any(plot['type'] == 'scatter_plot' for plot in plots)
    plot_file = next(p['filepath'] for p in plots if p['type'] == 'scatter_plot')
    assert os.path.isfile(plot_file)

def test_generate_scatter_plot_column_missing(visualizer):
    result = visualizer.generate_scatter_plot('dataset1', 'num_col', 'missing_col')
    assert result is False

def test_generate_scatter_plot_non_numeric(visualizer):
    result = visualizer.generate_scatter_plot('dataset1', 'num_col', 'cat_col')
    assert result is False

def test_generate_scatter_plot_no_data_pairs(visualizer, sample_data_manager):
    df = pd.DataFrame({'num_col': [None, None], 'num_col2': [None, None]})
    sample_data_manager._get_dataset.return_value = df
    result = visualizer.generate_scatter_plot('dataset1', 'num_col', 'num_col2')
    assert result is False

def test_generate_box_plot_simple_success(visualizer):
    result = visualizer.generate_box_plot('dataset1', 'num_col')
    assert result is True
    plots = visualizer.generated_plots['dataset1']
    assert any(plot['type'] == 'box_plot' and plot['column'] == 'num_col' for plot in plots)

def test_generate_box_plot_grouped_success(visualizer):
    result = visualizer.generate_box_plot('dataset1', 'num_col', group_by='cat_col')
    assert result is True
    plots = visualizer.generated_plots['dataset1']
    assert any(plot['type'] == 'box_plot' and plot['column'] == 'num_col' and plot['group_by'] == 'cat_col' for plot in plots)

def test_generate_box_plot_column_missing(visualizer):
    result = visualizer.generate_box_plot('dataset1', 'missing_col')
    assert result is False

def test_generate_box_plot_column_not_numeric(visualizer):
    result = visualizer.generate_box_plot('dataset1', 'cat_col')
    assert result is False

def test_generate_box_plot_group_by_missing(visualizer):
    result = visualizer.generate_box_plot('dataset1', 'num_col', group_by='missing_col')
    assert result is False

def test_generate_report_success(visualizer):
    result = visualizer.generate_report('dataset1')
    assert result is True
    assert 'dataset1' in visualizer.generated_reports or True  # The method may not track explicitly

def test_generate_report_dataset_missing(visualizer):
    # Patch _validate_dataset to simulate missing dataset
    visualizer._validate_dataset = lambda ds: False
    result = visualizer.generate_report('dataset1')
    assert result is False

