import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.adafedgcdr.base import BaseConfig, load_base, save_base, train_base
from model.adafedgcdr.data import Event, build_split, manifest_sha256
from model.fedgcdr.party import Client

TARGET = "Books"
SEED = 2026


def build_synthetic_events(users=100):
    events = []
    for user in range(users):
        user_id = f"u{user}"
        events.append(Event(user_id, "CDs", f"s{user % 30}", 0))
        if user % 2 == 0:
            events.append(Event(user_id, "Movies", f"m{user % 17}", 1))
        for index in range(12):
            events.append(Event(user_id, TARGET, f"t{(user * 13 + index) % 200}", 1000000 + index))
    return events


def test_config():
    return BaseConfig(
        target_domain=TARGET,
        source_rounds=1,
        target_rounds=1,
        local_epoch=1,
        device="cpu",
        seed=SEED,
    )


def rewrite_bcd_target_events(events, split):
    others = set(split.users["fusion"]) | set(split.users["validation"]) | set(split.users["test"])
    per_user_index = {}
    rewritten = []
    for event in sorted(events, key=lambda e: (e.user_id, e.domain, e.timestamp_ms, e.item_id)):
        if event.user_id in others and event.domain == TARGET:
            index = per_user_index.get(event.user_id, 0)
            per_user_index[event.user_id] = index + 1
            if index == 0:
                rewritten.append(event)
            else:
                rewritten.append(Event(event.user_id, TARGET, f"leak{event.user_id}_{index}", event.timestamp_ms + 5000000))
        else:
            rewritten.append(event)
    return rewritten


def test_target_training_only_touches_base_users(monkeypatch):
    events = build_synthetic_events()
    split = build_split(events, TARGET, SEED)
    calls = []
    original_train_gat = Client.train_gat
    original_train = Client.train

    def spy_train_gat(self, domain_id, *args, **kwargs):
        calls.append((self.user_id, self.domain_names[domain_id]))
        return original_train_gat(self, domain_id, *args, **kwargs)

    def spy_train(self, domain_id, *args, **kwargs):
        calls.append((self.user_id, self.domain_names[domain_id]))
        return original_train(self, domain_id, *args, **kwargs)

    monkeypatch.setattr(Client, "train_gat", spy_train_gat)
    monkeypatch.setattr(Client, "train", spy_train)
    train_base(events, split, test_config())
    target_calls = {user_id for user_id, domain in calls if domain == TARGET}
    assert target_calls == set(split.users["base"])
    assert all(domain != TARGET or user_id in set(split.users["base"]) for user_id, domain in calls)


def test_bcd_target_events_do_not_leak_into_item_vectors():
    events = build_synthetic_events()
    split = build_split(events, TARGET, SEED)
    first = train_base(events, split, test_config())
    second = train_base(rewrite_bcd_target_events(events, split), split, test_config())
    assert first.target_items == second.target_items
    assert torch.equal(first.target_item_vectors, second.target_item_vectors)
    others = set(split.users["fusion"]) | set(split.users["validation"]) | set(split.users["test"])
    assert not (set(first.target_train_users) & others)
    assert set(first.target_train_users) <= set(split.users["base"])
    assert set(first.target_train_users) == set(second.target_train_users)


def test_train_base_is_reproducible():
    events = build_synthetic_events()
    split = build_split(events, TARGET, SEED)
    first = train_base(events, split, test_config())
    second = train_base(events, split, test_config())
    assert first.domain_names == second.domain_names
    assert first.split_sha256 == manifest_sha256(split)
    assert first.split_sha256 == second.split_sha256
    assert torch.equal(first.target_item_vectors, second.target_item_vectors)
    assert set(first.source_vectors) == set(second.source_vectors)
    for user_id in first.source_vectors:
        assert torch.equal(first.source_vectors[user_id], second.source_vectors[user_id])
        assert torch.equal(first.source_counts[user_id], second.source_counts[user_id])
        assert torch.equal(first.source_mask[user_id], second.source_mask[user_id])


def test_export_covers_expected_users_masks_and_catalog():
    events = build_synthetic_events()
    split = build_split(events, TARGET, SEED)
    artifacts = train_base(events, split, test_config())
    user_ids = {event.user_id for event in events}
    source_names = sorted({event.domain for event in events if event.domain != TARGET})
    assert artifacts.domain_names == source_names
    assert set(artifacts.source_vectors) == user_ids
    assert set(artifacts.source_counts) == user_ids
    assert set(artifacts.source_mask) == user_ids
    dimension = test_config().embedding_size
    for user_id in user_ids:
        assert artifacts.source_vectors[user_id].shape == (len(source_names), dimension)
        assert artifacts.source_counts[user_id].shape == (len(source_names),)
        assert artifacts.source_mask[user_id].shape == (len(source_names),)
        assert torch.all(artifacts.source_vectors[user_id][~artifacts.source_mask[user_id]] == 0)
    assert artifacts.target_items
    assert artifacts.target_item_vectors.shape == (len(artifacts.target_items), dimension)
    assert list(artifacts.target_items) == sorted(artifacts.target_items)
    base_users_with_target = {
        event.user_id for event in events if event.domain == TARGET and event.user_id in set(split.users["base"])
    }
    assert set(artifacts.target_train_users) == base_users_with_target
    odd_user = next(uid for uid in sorted(user_ids) if int(uid[1:]) % 2 == 1)
    assert artifacts.source_mask[odd_user].tolist() == [True, False]


def test_save_and_load_roundtrip(tmp_path):
    events = build_synthetic_events()
    split = build_split(events, TARGET, SEED)
    artifacts = train_base(events, split, test_config())
    path = tmp_path / "base.pt"
    save_base(path, artifacts, test_config(), "events-digest")
    loaded, config, events_sha256 = load_base(path)
    assert events_sha256 == "events-digest"
    assert config == test_config()
    assert loaded.domain_names == artifacts.domain_names
    assert loaded.split_sha256 == artifacts.split_sha256
    assert loaded.target_items == artifacts.target_items
    assert loaded.target_train_users == artifacts.target_train_users
    assert torch.equal(loaded.target_item_vectors, artifacts.target_item_vectors)
    for user_id in artifacts.source_vectors:
        assert torch.equal(loaded.source_vectors[user_id], artifacts.source_vectors[user_id])
        assert torch.equal(loaded.source_counts[user_id], artifacts.source_counts[user_id])
        assert torch.equal(loaded.source_mask[user_id], artifacts.source_mask[user_id])
