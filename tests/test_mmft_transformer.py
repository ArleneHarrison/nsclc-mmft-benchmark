import unittest

import numpy as np
import pandas as pd
import torch

from scripts.train_mmft_transformer_petct import (
    MMFTTransformer,
    ModalityAwarePreprocessor,
    RiskTokenAppender,
    TabularBatch,
    initialize_prior_residual,
)


class MMFTTransformerTests(unittest.TestCase):
    def test_forward_returns_one_logit_per_row(self):
        batch = TabularBatch(
            x=torch.randn(7, 6),
            modality_ids=torch.tensor([0, 0, 1, 1, 2, 3]),
            feature_ids=torch.arange(6),
        )
        model = MMFTTransformer(
            n_features=6,
            n_modalities=4,
            d_model=16,
            n_heads=2,
            n_layers=1,
            dropout=0.1,
        )

        logits = model(batch)

        self.assertEqual(tuple(logits.shape), (7,))

    def test_preprocessor_selects_features_without_using_test_rows(self):
        train = pd.DataFrame(
            {
                "blood_a": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                "blood_b": [5.0, 5.1, 4.9, 5.0, 5.2, 4.8],
                "pet_a": [0.1, 0.2, 0.1, 0.2, 0.1, 0.2],
                "ct_a": [9.0, 8.0, 7.0, 6.0, 5.0, 4.0],
            }
        )
        test = pd.DataFrame(
            {
                "blood_a": [100.0, 101.0],
                "blood_b": [5.0, 5.0],
                "pet_a": [0.2, 0.1],
                "ct_a": [4.0, 9.0],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0, 1])
        modalities = {"blood": ["blood_a", "blood_b"], "pet": ["pet_a"], "ct": ["ct_a"]}
        pre = ModalityAwarePreprocessor(modalities=modalities, k=2)

        train_batch = pre.fit_transform(train, y)
        test_batch = pre.transform(test)

        self.assertEqual(train_batch.x.shape, (6, 2))
        self.assertEqual(test_batch.x.shape, (2, 2))
        self.assertIn("blood_a", pre.selected_features_)
        self.assertEqual(len(pre.selected_modalities_), 2)
        self.assertEqual(tuple(train_batch.modality_ids.shape), (2,))

    def test_risk_token_appender_adds_standardized_supervised_token(self):
        train_batch = TabularBatch(
            x=torch.tensor(
                [
                    [-1.0, 0.2],
                    [-0.8, 0.1],
                    [0.9, -0.1],
                    [1.1, -0.2],
                    [1.2, 0.0],
                    [-1.2, 0.3],
                ],
                dtype=torch.float32,
            ),
            modality_ids=torch.tensor([0, 1]),
            feature_ids=torch.tensor([0, 1]),
        )
        test_batch = TabularBatch(
            x=torch.tensor([[0.7, -0.1], [-0.7, 0.1]], dtype=torch.float32),
            modality_ids=torch.tensor([0, 1]),
            feature_ids=torch.tensor([0, 1]),
        )
        y = np.array([0, 0, 1, 1, 1, 0])

        appender = RiskTokenAppender(modality_id=2, c=1.0)
        new_train = appender.fit_transform(train_batch, y)
        new_test = appender.transform(test_batch)

        self.assertEqual(new_train.x.shape, (6, 3))
        self.assertEqual(new_test.x.shape, (2, 3))
        self.assertEqual(new_train.modality_ids[-1].item(), 2)
        self.assertAlmostEqual(float(new_train.x[:, -1].mean()), 0.0, places=5)

    def test_prior_residual_initialization_starts_from_risk_token(self):
        batch = TabularBatch(
            x=torch.tensor([[0.5, -1.0, 0.25], [1.0, 0.2, -0.75]], dtype=torch.float32),
            modality_ids=torch.tensor([0, 1, 2]),
            feature_ids=torch.tensor([0, 1, 2]),
        )
        model = MMFTTransformer(
            n_features=3,
            n_modalities=3,
            d_model=12,
            n_heads=3,
            n_layers=1,
            dropout=0.0,
        )

        initialize_prior_residual(model, risk_feature_index=2)
        logits = model(batch)

        np.testing.assert_allclose(logits.detach().numpy(), batch.x[:, 2].numpy(), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
