"""UI changes preserve the weather raster, map coordinates and location references."""
from datetime import datetime
from io import BytesIO
from pathlib import Path
import sys
from unittest.mock import Mock, patch

import numpy as np
import pytest
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from preview_weather_ui import METHODS, renderer


@pytest.mark.parametrize('title,badge', [
    ('Actual radar | 20 Oct 12:00 SGT', 'RADAR OBSERVATION'),
    ('Forecast (+5 min) | 20 Oct 12:05 SGT', 'FORECAST'),
])
def test_card_preserves_grid_coordinates_and_source_type(title, badge):
    functions = renderer(METHODS.read_text(encoding='utf-8'))
    grid = np.linspace(0, 1, 96).reshape(8, 12)
    markers = [('Main', '#e83288', (2, 3)), ('Office', '#0072b2', (8, 5))]
    image = None
    figure = None
    try:
        with patch.object(plt, 'close'):
            image = functions['_render_heatmap'](grid, title, 'Radar intensity',
                                                  markers, marker_numbers=[1, 3])
            figure = plt.gcf()
            axes = figure.axes[0]
            assert np.array_equal(axes.images[1].get_array(), grid)
            assert axes.images[1].norm(.25) == pytest.approx(.25 ** .6)
            from matplotlib.offsetbox import AnnotationBbox
            from matplotlib.text import Text
            pins = [item for item in axes.artists if isinstance(item, AnnotationBbox)]
            assert [pin.xy for pin in pins] == [(2, 3), (8, 5)]
            assert [child.get_text() for pin in pins for child in pin.offsetbox.get_children()
                    if isinstance(child, Text)] == ['1', '3']
            # The drawn tip must transform to the original raster coordinate.
            for pin in pins:
                tip = pin.offsetbox.get_transform().transform((11, 0))
                assert np.allclose(tip, axes.transData.transform(pin.xy))
            assert badge in [item.get_text() for item in figure.texts]
            assert image.read(8) == b'\x89PNG\r\n\x1a\n'
    finally:
        if image is not None:
            image.close()
        if figure is not None:
            plt.close(figure)


def test_outside_coverage_does_not_renumber_later_location_markers():
    functions = renderer(METHODS.read_text(encoding='utf-8'))
    rows = [dict(label='Main', latitude=1.3, longitude=103.8),
            dict(label='Outside', latitude=2., longitude=103.8),
            dict(label='Office', latitude=1.35, longitude=103.9)]
    draw = Mock(return_value=BytesIO())
    functions['render_heatmap'] = draw
    image, caption = functions['build_group_map'](rows, np.zeros((8, 12)), datetime(2026, 10, 20, 12), actual=True)
    image.close()
    assert draw.call_args.kwargs['marker_numbers'] == [1, 3]
    assert [item[0] for item in draw.call_args.kwargs['marker']] == ['Main', 'Office']
    assert '2. Outside: outside coverage' in caption
    assert '3. Office: No rain detected' in caption
