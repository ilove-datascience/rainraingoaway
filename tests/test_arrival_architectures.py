import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from arrival.models import ArrivalNet

class ArchitectureTests(unittest.TestCase):
    def test_spatial_change_head(self):
        model=ArrivalNet(pooling='spatial_change',recurrent_layers=2)
        x=torch.randn(2,6,7,32,32)
        loss=-model.log_probs(x)[:,0].mean();loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        model.eval();copy=ArrivalNet(**model.config);copy.load_state_dict(model.state_dict());copy.eval()
        torch.testing.assert_close(model(x),copy(x))

    def test_deeper_variants(self):
        torch.set_num_threads(2)
        for kind,depth,extra in (('gru',2,0),('gru',3,0),('gru',1,2),('lstm',2,0)):
            model=ArrivalNet(kind=kind,recurrent_layers=depth,encoder_extra_layers=extra,pooling='spatial')
            lp=model.log_probs(torch.randn(2,6,7,32,32))
            (-lp[:,0].mean()).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            clone=ArrivalNet(**model.config);clone.load_state_dict(model.state_dict())

    def test_variants_backward_and_roundtrip(self):
        torch.set_num_threads(2)
        for width,pooling in ((64,'global'),(96,'global'),(64,'spatial')):
            model=ArrivalNet(hidden_dim=width,pooling=pooling)
            x=torch.randn(2,6,7,64,64)
            lp=model.log_probs(x)
            self.assertEqual(tuple(lp.shape),(2,13))
            (-lp[:,0].mean()).backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            clone=ArrivalNet(**model.config);clone.load_state_dict(model.state_dict())
            model.eval();clone.eval()
            torch.testing.assert_close(model(x),clone(x))

    def test_old_configuration_loads(self):
        old=dict(history=6,channels=7,kind='gru',output='categorical',dropout=.15)
        model=ArrivalNet(**old)
        self.assertEqual(model.cell.hidden_dim,64)
        self.assertEqual(model.head[3].in_features,64)

if __name__=='__main__':unittest.main()
