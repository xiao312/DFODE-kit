"""Synthetic scorer tests, not a reproduction of private chemistry experiments."""
import json
import unittest
import numpy as np
from benchmarks.research_contract.reference import evaluate_predictions

class MetricContractTests(unittest.TestCase):
    def score(self,p,d,b=None,**kwargs):
        return evaluate_predictions(np.array(p,float).reshape(-1,1),np.array(d,float).reshape(-1,1),
            np.array(d if b is None else b,float).reshape(-1,1),['X'],**kwargs)

    def test_exact_prediction(self):
        r=self.score([0,1e-32,1e-12,-1e-3],[0,1e-32,1e-12,-1e-3])
        self.assertEqual(r['groups']['all']['acceptance'],1)
        self.assertEqual(r['all_active_species_state_acceptance'],1)

    def test_motivating_overprediction(self):
        r=self.score([1e-8],[1e-12])
        self.assertAlmostEqual(r['groups']['all']['normalized_error']['max'],99000)
        e=next(q for q in r['small_target_events'] if q['epsilon']==1e-12 and q['eta']==1e-8)
        self.assertEqual(e['events'],1)
        self.assertFalse(e['tenfold_reduction_assessable'])

    def test_zero_predictor_fails_nontrivial(self):
        r=self.score([0,0,0],[0,1e-12,-1e-3])
        self.assertEqual(r['groups']['nontrivial']['acceptance'],0)
        self.assertEqual(r['nontrivial_sign']['wrong_sign_count'],2)

    def test_wrong_sign_fails(self):
        r=self.score([-1e-12],[1e-12])
        self.assertEqual(r['groups']['all']['pass_count'],0)
        self.assertEqual(r['nontrivial_sign']['wrong_sign_count'],1)
        self.assertEqual(r['same_sign_decade_error']['components'],0)

    def test_inclusive_boundary(self):
        self.assertEqual(self.score([1e-15],[0])['groups']['all']['acceptance'],1)
        self.assertEqual(self.score([np.nextafter(1e-15,np.inf)],[0])['groups']['all']['acceptance'],0)

    def test_repair_damage_identity(self):
        r=self.score([1,2,1,2],[1,1,1,1],[1,1,2,2])['groups']['all']
        self.assertEqual(r['repaired'],1); self.assertEqual(r['damaged'],1)
        self.assertEqual(r['repair_rate_among_base_failures'],.5)
        self.assertEqual(r['damage_rate_among_base_successes'],.5)
        self.assertEqual(r['acceptance'],r['base_acceptance']+r['acceptance_gain'])

    def test_empty_denominators_are_null(self):
        r=self.score([0],[0])
        self.assertIsNone(r['groups']['nontrivial']['acceptance'])
        self.assertIsNone(r['groups']['all']['repair_rate_among_base_failures'])
        self.assertIsNone(r['nontrivial_sign']['wrong_sign_rate'])

    def test_nonfinite_candidate_not_dropped(self):
        r=self.score([np.nan,1e-3],[1e-3,1e-3])
        self.assertFalse(r['returned_prediction_finite'])
        self.assertEqual(r['groups']['all']['components'],2)
        self.assertEqual(r['groups']['all']['acceptance'],.5)
        self.assertIsNone(r['groups']['all']['normalized_error']['p99'])
        json.dumps(r,allow_nan=False)

    def test_nonfinite_reference_rejected(self):
        with self.assertRaises(ValueError): self.score([0],[np.nan])

    def test_nonfinite_baseline_rejected(self):
        with self.assertRaises(ValueError): self.score([0],[0],[np.inf])

    def test_species_shape_required(self):
        with self.assertRaises(ValueError):
            evaluate_predictions([[1]],[[1]],[[1]],['A','B'])

    def test_duplicate_species_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_predictions([[1,1]],[[1,1]],[[1,1]],['A','A'])

    def test_argon_excluded_but_validated(self):
        r=evaluate_predictions([[1,1]],[[1,0]],[[1,0]],['X','AR'])
        self.assertEqual(r['groups']['all']['components'],1)
        self.assertEqual(r['argon_nonzero_or_nonfinite_returned_count'],1)

    def test_qualified_mask_is_candidate_independent(self):
        u=np.array([[1e-18],[np.nan]])
        a=self.score([1e-12,1e-6],[1e-12,1e-6],reference_uncertainty=u)
        b=self.score([1,1],[1e-12,1e-6],reference_uncertainty=u)
        self.assertEqual(a['qualified']['coverage'],.5)
        self.assertEqual(a['qualified']['all']['components'],b['qualified']['all']['components'])

    def test_negative_uncertainty_rejected(self):
        with self.assertRaises(ValueError):
            self.score([0],[0],reference_uncertainty=[[-1]])

    def test_sign_stability_qualification(self):
        r=self.score([-1e-12,1e-6],[1e-12,1e-6],reference_uncertainty=[[1e-18],[1e-18]],
            reference_sign_stable=np.array([[True],[False]]))
        self.assertEqual(r['qualified']['sign']['components'],1)
        self.assertEqual(r['qualified']['sign']['wrong_sign_count'],1)

    def test_negative_endpoint_counts(self):
        r=self.score([-2e-6],[0],initial=[[1e-6]])
        self.assertEqual(r['endpoint_diagnostics']['negative_components'],1)

    def test_primary_groups_partition(self):
        d=[0,1e-20,1e-15,1.2e-15,1e-12,2e-12,1e-3]
        r=self.score(d,d)
        self.assertEqual(sum(r['groups'][g]['components'] for g in
            ['floor_dominated','weak_nontrivial','active_increment']),len(d))

    def test_same_sign_magnitude_error(self):
        r=self.score([1e-8],[1e-12])
        self.assertAlmostEqual(r['same_sign_decade_error']['absolute_log10_ratio']['max'],4.)

    def test_asinh_sufficient_bound_algebra(self):
        d=np.r_[0,np.logspace(-32,-3,200),-np.logspace(-32,-3,200)]
        s=1e-14; delta=np.log1p(.1)*(1-1e-10)
        for sign in [-1,1]:
            p=s*np.sinh(np.arcsinh(d/s)+sign*delta)
            r=self.score(p,d)
            self.assertEqual(r['groups']['all']['acceptance'],1.)

    def test_invalid_tolerances(self):
        for a,r in [(0,.1),(1e-15,1),(np.nan,.1),(1e-15,-1)]:
            with self.assertRaises(ValueError):self.score([0],[0],atol=a,rtol=r)

    def test_empty_arrays_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_predictions(np.empty((0,1)),np.empty((0,1)),np.empty((0,1)),['X'])

if __name__=='__main__':
    unittest.main()
