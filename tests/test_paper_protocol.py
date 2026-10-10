"""Runs with unittest as well as pytest; verifies actual leakage boundaries."""
import unittest
from model.adafedgcdr.data import Event, SplitManifest, user_views
from model.adafedgcdr.paper import prefix_cases, target_inputs, quarantine_development_users
from model.adafedgcdr.paper import popularity_records, prepare_rows
from model.adafedgcdr.base import BaseArtifacts
from model.adafedgcdr.experiment import Catalog, fuse_user_vec
from model.adafedgcdr.fusion import AdaptiveFusion
import torch


class PaperProtocolTests(unittest.TestCase):
    def setUp(self):
        self.manifest=SplitManifest('Books',2026,{'base':('a',),'fusion':('b',),'validation':('c',),'test':('d',)})
        self.events=[]
        for user in 'abcd':
            self.events.append(Event(user,'CDs','s',0))
            self.events.extend(Event(user,'Books',f't{i}',100+i) for i in range(12))

    def test_sparse_only_exposes_validation_test_prefix(self):
        _,visible,_=target_inputs(user_views(self.events,'Books'),self.manifest,'sparse',3)
        self.assertEqual(visible['c'],('t0','t1','t2'))
        self.assertEqual(visible['d'],('t0','t1','t2'))
        self.assertEqual(len(visible['a']),12)
        self.assertEqual(len(visible['b']),12)

    def test_cold_only_base_users_enter_target_training(self):
        _,visible,train=target_inputs(user_views(self.events,'Books'),self.manifest,'cold')
        self.assertEqual(train,{'a'})
        self.assertFalse(visible['b'] or visible['c'] or visible['d'])

    def test_future_changes_do_not_change_sparse_training_input_or_catalog(self):
        changed=[Event(e.user_id,e.domain,'future',e.timestamp_ms) if e.user_id in 'cd' and e.domain=='Books' and e.timestamp_ms>=103 else e for e in self.events]
        before=target_inputs(user_views(self.events,'Books'),self.manifest,'sparse',3)
        after=target_inputs(user_views(changed,'Books'),self.manifest,'sparse',3)
        self.assertEqual(before,after)

    def test_cold_fusion_labels_cannot_build_item_catalog(self):
        changed=[Event(e.user_id,e.domain,'new-training-b-item',e.timestamp_ms) if e.user_id=='b' and e.domain=='Books' else e for e in self.events]
        before=target_inputs(user_views(self.events,'Books'),self.manifest,'cold')[0]
        after=target_inputs(user_views(changed,'Books'),self.manifest,'cold')[0]
        self.assertEqual(before,after)

    def test_fixed_label_is_identical_across_history_sizes(self):
        cases=prefix_cases(self.events,self.manifest,'test',[0,1,3,5,10],True)
        self.assertEqual({c.label_item_id for c in cases},{'t10'})
        self.assertEqual([len(c.history_item_ids) for c in cases],[0,1,3,5,10])
        self.assertTrue(all(c.label_item_id not in c.history_item_ids for c in cases))

    def test_zero_prefix_rejected_by_sparse_protocol(self):
        with self.assertRaises(ValueError):
            target_inputs(user_views(self.events,'Books'),self.manifest,'sparse',0)

    def test_development_test_users_are_quarantined(self):
        new,moved=quarantine_development_users(self.manifest,['d'])
        self.assertEqual(moved,('d',))
        self.assertEqual(new.users['test'],())
        self.assertEqual(new.users['validation'],('c','d'))
        self.assertEqual(new.users['base'],self.manifest.users['base'])
        self.assertEqual(new.users['fusion'],self.manifest.users['fusion'])
        groups=[set(v) for v in new.users.values()]
        self.assertFalse(any(groups[i]&groups[j] for i in range(4) for j in range(i+1,4)))

    def fixtures(self):
        items={f't{i}':i for i in range(200)}
        base=BaseArtifacts(['CDs','Clothing','Movies'],{'d':torch.zeros(3,16)},
            {'d':torch.ones(3)},{'d':torch.ones(3,dtype=torch.bool)},items,
            torch.zeros(200,16),['a','b'],'split')
        counts=dict.fromkeys(items,0)
        counts['t1']=1000
        catalog=Catalog(tuple(items),{'d':frozenset(f't{i}' for i in range(12))},counts)
        return base,catalog

    def test_unknown_prefix_is_excluded_for_both_sparse_methods(self):
        changed=[Event(e.user_id,e.domain,'unknown',e.timestamp_ms) if e.user_id=='d' and e.domain=='Books' and e.timestamp_ms==100 else e for e in self.events]
        base,catalog=self.fixtures()
        rows,audit=prepare_rows(prefix_cases(changed,self.manifest,'test',[1]),base,catalog,True)
        self.assertEqual(rows,[])
        self.assertEqual(audit['no_known_prefix'],1)
        self.assertEqual(audit['unknown_label'],0)

    def test_popularity_baseline_uses_training_frequency(self):
        base,catalog=self.fixtures()
        rows,_=prepare_rows(prefix_cases(self.events,self.manifest,'test',[1]),base,catalog,True)
        records=popularity_records(rows,'cpu')
        self.assertEqual(len(records),1)
        self.assertEqual(records[0]['ndcg@10'],1.)

    def test_target_only_cannot_use_sources_when_history_is_unknown(self):
        batch={'source_vec':torch.ones(1,3,16),'source_mask':torch.ones(1,3,dtype=torch.bool),
            'source_count':torch.ones(1,3),'history_vec':torch.zeros(1,1,16),
            'history_mask':torch.zeros(1,1,dtype=torch.bool)}
        user,personalized=fuse_user_vec('target-only',AdaptiveFusion(3),batch)
        self.assertTrue(torch.equal(user,torch.zeros_like(user)))
        self.assertFalse(personalized.item())

    def test_source_only_cannot_use_target_history_without_sources(self):
        batch={'source_vec':torch.zeros(1,3,16),'source_mask':torch.zeros(1,3,dtype=torch.bool),
            'source_count':torch.zeros(1,3),'history_vec':torch.ones(1,1,16),
            'history_mask':torch.ones(1,1,dtype=torch.bool)}
        user,personalized=fuse_user_vec('source-only',AdaptiveFusion(3),batch)
        self.assertTrue(torch.equal(user,torch.zeros_like(user)))
        self.assertFalse(personalized.item())


if __name__=='__main__':
    unittest.main()
