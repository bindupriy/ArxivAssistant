"""Shared synchronization of config-defined research domains."""

from sqlalchemy import select

from assistant.config import get_config
from assistant.storage import Domain, session_scope
from assistant.storage.index_lock import resource_lock


def sync_domains_from_config() -> tuple[int, int]:
    with resource_lock("domains-sync"):
        return _sync_domains()


def _sync_domains() -> tuple[int, int]:
    added = 0
    updated = 0
    with session_scope() as session:
        for domain in get_config().domains:
            row = session.execute(
                select(Domain).where(Domain.name == domain.name)
            ).scalar_one_or_none()
            if row is None:
                session.add(
                    Domain(
                        name=domain.name,
                        arxiv_categories=list(domain.arxiv_categories),
                        venues=list(domain.venues),
                        seed_papers=list(domain.seed_papers),
                    )
                )
                added += 1
            else:
                row.arxiv_categories = list(domain.arxiv_categories)
                row.venues = list(domain.venues)
                row.seed_papers = list(domain.seed_papers)
                updated += 1
    return added, updated