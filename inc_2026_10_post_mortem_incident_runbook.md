# INC-2026-10: Engineering Incident Post-Mortems & Episodic Runbook

* **Document ID:** `doc_inc_runbook`
* **Document Type:** Operational Post-Mortem & Episodic History
* **On-Call SRE:** Rohan Gupta (`rohan@company.internal`)
* **On-Call Backend Lead:** Priya Sharma (`priya@company.internal`)
* **Affected Services:** `AuthService`, `AuthRedisCache`, `TwilioGateway`

---

## Incident Episode 1: `ep_oct14_504` (Oct 14, 2026)
* **Session ID:** `sess_inc_104`
* **Affected Services:** `AuthService`, `AuthRedisCache`
* **Symptom:** Users experienced intermittent `504 Gateway Timeout` errors on `/auth/v1/token` during peak login traffic.
* **Root Cause:** `AuthRedisCache` connection pool inside `AuthService` was hardcoded to a maximum of 50 connections, leading to thread starvation when concurrent token blacklist checks spiked.
* **Verified Resolution:** Priya Sharma and Rohan Gupta scaled `max_connections` from `50` to `250` in the `AuthService` production config map and enabled idle connection eviction.

---

## Incident Episode 2: `ep_oct08_sms` (Oct 08, 2026)
* **Session ID:** `sess_inc_092`
* **Affected Services:** `TwilioGateway`, `PhoneSignup`
* **Symptom:** SMS OTP delivery failures across international dial codes (`+91`, `+44`), causing a 38% drop in `PhoneSignup` completions.
* **Root Cause:** `TwilioGateway` hit the sandbox throughput rate limit on the secondary carrier route.
* **Verified Resolution:** Rohan Gupta upgraded the `TwilioGateway` routing profile to the Enterprise high-throughput messaging pool and added automatic retry backoff.