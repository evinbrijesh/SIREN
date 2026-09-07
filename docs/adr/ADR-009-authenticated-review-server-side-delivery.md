# ADR-009 — Authenticated Review and Server-Side Delivery

**Status:** Proposed · **Date:** 2026-09-07 · **Applies to:** review workflow and alert dispatch (hosted operation)

## Context

The hackathon implementation has two gaps that must be closed for a live hazard service:

**Authentication gap.** The reviewer identity (`coordinator-01`) is client-supplied with no verification. Any HTTP client can POST a `confirm` decision for any run. There is no access control separating who can trigger processing, who can review, and who can dispatch.

**Delivery gap.** The backend dispatch endpoint writes `"status": "sent"` without contacting any external provider. Live alert delivery is handled entirely by a browser-side ntfy.sh call in `frontend/src/utils/ntfy.ts`. This means:
- Closing the browser tab cancels unsent notifications.
- A network error during the ntfy POST is treated as success if `navigator.onLine` was true.
- `"sent"` in the database does not mean delivered.
- No receipts, retries, or expiration are recorded.
- The public ntfy topic (`siren-emergency-alert`) is unmoderated and publicly subscribable.

The PRD's human gate (hard rule 3) requires a confirmed human decision before any alert leaves the system. This ADR extends that gate to cover identity, validity, and verified delivery — without removing the human decision requirement.

## Decision

### Authentication and authorization

Use OIDC-compatible authentication (Cognito, organizational IdP, or equivalent) for the hosted service.

Define three roles with separate permissions:

| Role | Can do |
|---|---|
| `ingestion-worker` | Register observations, create acquisition jobs, trigger processing |
| `coordinator` | View runs, submit reviews (confirm/reject/postpone), view dispatches |
| `admin` | All of the above plus basin config, recipient management, audit export |

**`ingestion-worker` cannot submit reviews or trigger dispatch.** The account that downloads satellite imagery cannot also confirm an alert. This separation is the structural expression of the human gate.

Reviewer identity in review records must come from authenticated claims, not from the POST body. The current `"reviewer": "coordinator-01"` in the request body is replaced by the authenticated principal.

### Review validity and expiry

A review is valid only when:
1. It applies to the most recently completed run for the observation (not a superseded run).
2. It is the **latest** review decision for that run (a later `reject` or `postpone` suppresses a prior `confirm`).
3. It was recorded within the configured validity window (e.g. 4 hours from run completion). Stale confirms for hours-old assessments must not drive dispatch of a current alert.

The current implementation allows any historical `confirm` to authorize dispatch, including one that was subsequently rejected or postponed. This must be fixed.

### Server-side delivery outbox

Replace the browser-side ntfy call with a server-side transactional delivery outbox:

```sql
CREATE TABLE delivery_jobs (
    delivery_id     TEXT PRIMARY KEY,
    dispatch_id     TEXT NOT NULL REFERENCES dispatches(dispatch_id),
    provider        TEXT NOT NULL,    -- 'ntfy' | 'sms' | 'lora' | 'satellite-sbd'
    endpoint        TEXT NOT NULL,
    payload_bytes   INTEGER NOT NULL,
    status          TEXT NOT NULL,    -- pending | attempted | delivered | failed | expired
    provider_receipt TEXT,
    attempts        INTEGER DEFAULT 0,
    last_error      TEXT,
    next_retry_at   TEXT,
    expires_at      TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
```

When a coordinator confirms an alert:
1. The review is recorded with the authenticated identity.
2. The dispatch record is created.
3. A delivery job is inserted in the same transaction (transactional outbox pattern).
4. A background worker processes the delivery job: calls the provider, records the receipt, updates the status.
5. If delivery fails, the worker retries with bounded backoff until the `expires_at` time.
6. `"delivered"` in the database means the provider accepted the message, not that the recipient read it.

The frontend may still show a confirmation UI and log the decision locally, but it is never the authoritative delivery path.

### Simulation vs. live delivery modes

The demo/simulation deployment retains simulated dispatch (no real provider calls) per ADR-004. The hosted deployment uses the delivery outbox with approved provider credentials.

The frontend UI does not know which mode is active — it polls `GET /dispatches/{dispatch_id}` for status. The backend controls which mode is configured.

## Consequences

- **Positive:** reviewer identity is verifiable, not client-asserted.
- **Positive:** a later reject/postpone correctly suppresses an undelivered dispatch.
- **Positive:** delivery is durable — server restart does not lose a pending alert.
- **Positive:** delivery receipts and expiration are auditable.
- **Positive:** separation of roles prevents an ingestion service from autonomously dispatching alerts.
- **Negative:** requires OIDC/auth infrastructure not present in the demo.
- **Negative:** adds delivery job table and background worker to the operational service.
- **Negative:** server-side delivery requires approved provider credentials, which vary by country and alerting authority.
- **No autonomous dispatch is introduced.** Every delivery job is created only after a recorded confirmed human review. The human gate is extended, not removed.

## Rationale

A hazard alert system that writes `"sent"` for a browser-side HTTP call that may have been lost is not operationally trustworthy. Authenticated identity is a prerequisite for accountability. These are not sophistication add-ons — they are the minimum requirements for a system that claims to influence real emergency decisions.

The ntfy.sh integration is retained for demo and field-testing purposes. It must be clearly labeled as a testing channel, not as the operational delivery mechanism.

## Related decisions

- ADR-006: separation of acquisition and execution; the ingestion-worker role formalized here.
- ADR-007: PostgreSQL that provides the transaction isolation for the transactional outbox pattern.
- ADR-008: durable job pattern that the delivery outbox follows.
