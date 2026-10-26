"""Data-free tests of the model's scientific contracts."""
import io
import unittest

import torch

from pgfm.models.flow import PiecewisePhysFlowMatchingUNet
from pgfm.training import PiecewiseFlowMatchingPhysicsTrainer


def make_model(**kwargs):
    return PiecewisePhysFlowMatchingUNet(
        past_hours=2, bg_scalar_channels=0, bg_map_channels=0,
        tec_std=10.0, hidden=8, **kwargs,
    )


def inputs():
    return dict(
        x_tau=torch.randn(2, 1, 16, 16),
        tau=torch.tensor([0.1, 0.7]).reshape(2, 1, 1, 1),
        tec_hist=torch.randn(2, 2, 16, 16),
        bg_scalar_hist=torch.empty(2, 0, 2),
        bg_map_hist=torch.empty(2, 0, 2, 16, 16),
        future_terms_hour=torch.randn(2, 4, 4, 16, 16) * 1e-4,
        future_uv_hour=torch.randn(2, 2, 4, 16, 16),
        future_rhs_hour=torch.randn(2, 1, 4, 16, 16) * 1e-4,
    )


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(17)

    def test_mixing_convention(self):
        model = make_model()
        model.lambda_head.weight.data.zero_()
        model.lambda_head.bias.data.fill_(2.0)
        u, aux = model(**inputs(), return_aux=True)
        expected = torch.full_like(u, 1 / (1 + torch.exp(torch.tensor(2.0))))
        torch.testing.assert_close(aux['lambda_map'], expected)
        torch.testing.assert_close(u, expected * aux['u_data'] + (1-expected) * aux['u_phys'])

    def test_piecewise_target_boundaries(self):
        trainer = PiecewiseFlowMatchingPhysicsTrainer(make_model(), use_amp=False)
        knots = torch.tensor([0., 1., 3., 6., 10.]).view(1, 5, 1, 1).repeat(5, 1, 1, 1)
        tau = torch.tensor([0., .25, .5, .75, 1.]).view(5, 1, 1, 1)
        x, u = trainer._build_piecewise_target(knots, tau)
        torch.testing.assert_close(x.flatten(), torch.tensor([0., 1., 3., 6., 10.]))
        torch.testing.assert_close(u.flatten(), torch.tensor([4., 8., 12., 16., 16.]))

    def test_rhs_units_and_gate_bypass(self):
        model = make_model()
        data = inputs()
        data['future_rhs_hour'] = torch.ones_like(data['future_rhs_hour']) * 0.01
        _, aux = model(**data, return_aux=True)
        torch.testing.assert_close(aux['u_phys'], torch.full_like(aux['u_phys'], 3.6))
        aux['u_phys'].sum().backward()
        for param in model.uv_gate.parameters():
            self.assertIsNotNone(param.grad)
            self.assertEqual(torch.count_nonzero(param.grad).item(), 0)

    def test_lambda_receives_gradient_and_optimizer_updates(self):
        model = make_model()
        trainer = PiecewiseFlowMatchingPhysicsTrainer(model, use_amp=False)
        data = inputs()
        batch = {k: data[k] for k in ('tec_hist', 'bg_scalar_hist', 'bg_map_hist')}
        batch.update(knots_hour1=torch.randn(2, 5, 16, 16),
                     future_terms_hour1=data['future_terms_hour'],
                     future_uv_hour1=data['future_uv_hour'],
                     future_rhs_hour1=data['future_rhs_hour'])
        before = model.lambda_head.weight.detach().clone()
        loss, metrics = trainer._compute_loss(batch)
        torch.testing.assert_close(loss.detach(), metrics['loss_main'] + 1e-4 * metrics['loss_balance'] + 1e-4 * metrics['loss_phys'])
        loss.backward()
        self.assertTrue(torch.isfinite(model.lambda_head.weight.grad).all())
        self.assertGreater(model.lambda_head.weight.grad.abs().sum().item(), 0)
        trainer.opt.step()
        self.assertFalse(torch.equal(before, model.lambda_head.weight))

    def test_configurable_empty_context_widths(self):
        u = make_model(bg_map_proj_ch=4, bg_scalar_proj_ch=3)(**inputs())
        self.assertEqual(u.shape, (2, 1, 16, 16))
        self.assertTrue(torch.isfinite(u).all())

    def test_checkpoint_roundtrip(self):
        model = make_model().eval()
        data = inputs()
        expected = model(**data)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = make_model().eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        torch.testing.assert_close(expected, restored(**data), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
