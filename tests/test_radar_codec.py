import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from data_processing.radar_codec import SOURCE,LEGACY,SOURCE_HEX,SOURCE_CATEGORIES,decode_png,source_category
from data_processing.pngtojson import png_to_xy_intensity,points_to_intensity_grid
from data_processing.data_loading import _get_cache_paths,_read_cache,_write_cache
from data_processing.model_contract import write_contract,validate_contract


class RadarCodecTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.path=self.root/'image.png'
        colors=[tuple(int(c[i:i+2],16) for i in (0,2,4))+(255,) for c in SOURCE_HEX]
        Image.fromarray(np.array([colors],dtype=np.uint8),'RGBA').save(self.path)

    def test_source_levels_and_categories(self):
        values=decode_png(self.path,SOURCE)[0]
        self.assertEqual(len(np.unique(values)),33)
        self.assertTrue(np.all(np.diff(values)>0))
        self.assertEqual([source_category(float(v)) for v in values],list(SOURCE_CATEGORIES))
        self.assertEqual(source_category(float(values[10])),'Light')

    def test_legacy_is_bitwise_unchanged(self):
        original=np.asarray(points_to_intensity_grid(png_to_xy_intensity(self.path,True)),dtype=np.float32)/100
        np.testing.assert_array_equal(decode_png(self.path),original)

    def test_unknown_opaque_rejected_transparent_ignored(self):
        Image.fromarray(np.array([[[1,2,3,0],[0,255,255,255]]],dtype=np.uint8),'RGBA').save(self.path)
        self.assertEqual(decode_png(self.path,SOURCE)[0,0],0)
        Image.fromarray(np.array([[[1,2,3,255]]],dtype=np.uint8),'RGBA').save(self.path)
        with self.assertRaisesRegex(ValueError,'Unknown opaque'):decode_png(self.path,SOURCE)

    def test_versioned_cache_and_legacy_metadata(self):
        _,old,oldmeta=_get_cache_paths(self.root,'tick',LEGACY)
        _,new,newmeta=_get_cache_paths(self.root,'tick',SOURCE)
        self.assertNotEqual(old,new)
        array=np.zeros((7,2,2),dtype=np.float32)
        _write_cache(old,oldmeta,self.path,self.path,array,LEGACY)
        meta=json.loads(oldmeta.read_text());meta.pop('decoder_version');oldmeta.write_text(json.dumps(meta))
        self.assertIsNotNone(_read_cache(old,oldmeta,None,None,LEGACY))
        self.assertIsNone(_read_cache(old,oldmeta,None,None,SOURCE))
        _write_cache(new,newmeta,self.path,self.path,array,SOURCE)
        self.assertIsNotNone(_read_cache(new,newmeta,None,None,SOURCE))
        self.assertIsNone(_read_cache(new,newmeta,None,None,LEGACY))

    def test_model_contract_prevents_mixing(self):
        model=self.root/'model.pkl';model.write_bytes(b'checkpoint')
        norm=self.root/'norm.json';norm.write_text('{}')
        self.assertIsNone(validate_contract(model))
        with self.assertRaises(ValueError):validate_contract(model,SOURCE)
        with self.assertRaises(ValueError):write_contract(model,norm,SOURCE)
        norm.write_text(json.dumps({'radar_decoder_version':SOURCE}))
        write_contract(model,norm,SOURCE)
        validate_contract(model,SOURCE,norm)
        with self.assertRaises(ValueError):validate_contract(model,LEGACY)
        norm.write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError,'Normalization'):validate_contract(model,SOURCE,norm)

    def test_actual_caption_uses_official_category(self):
        import ast
        from io import BytesIO
        from datetime import datetime
        from telegram_code.local_rain import local_rain, in_coverage
        from telegram_code.notification_text import rain_notice
        from data_processing.data_loading import remove_small_echoes
        source=(Path(__file__).resolve().parents[1]/'src/telegram_code/methods.py').read_text(encoding='utf-8')
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='build_radar_snapshot_plot')
        env=dict(decode_png=decode_png,SOURCE=SOURCE,source_category=source_category,np=np,
                 remove_small_echoes=remove_small_echoes,local_rain=local_rain,in_coverage=in_coverage,
                 datetime=datetime,rain_notice=rain_notice,render_heatmap=lambda *a,**k:BytesIO())
        exec(compile(ast.Module(body=[node],type_ignores=[]),'actual-caption','exec'),env)
        path=self.root/'202608101200.png'
        image=np.zeros((120,217,4),dtype=np.uint8);image[:]=[0,202,17,255]
        Image.fromarray(image,'RGBA').save(path)
        buffer,caption=env['build_radar_snapshot_plot'](path,(1.3,103.8))
        self.addCleanup(buffer.close)
        self.assertIn('Observed intensity: Light (source radar scale)',caption)
