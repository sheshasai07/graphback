# RFC-012: Authentication & Token Lifecycle Architecture

* **Document ID:** `doc_rfc_012`
* **Document Type:** Architecture Specification
* **Status:** Active (Production)
* **Primary Maintainer:** Priya Sharma (`priya@company.internal`, Staff Backend Engineer)
* **Secondary SRE Contact:** Rohan Gupta (`rohan@company.internal`, Senior SRE & DevOps)
* **Target Systems:** `AuthService`, `TwilioGateway`, `AuthRedisCache`, `FraudCheckService`

---

## 1. Overview
`AuthService` (written in Go, Tier-0 criticality) is the central identity and session management microservice for all user-facing applications. It handles user registration, credential validation, JWT access/refresh token issuance, and Multi-Factor Authentication (MFA) orchestration.

## 2. System Dependencies & Call Graph
Whenever a user signs up or logs in via the `PhoneSignup` feature, `AuthService` executes the following synchronous dependency chain:

1. **Token Blacklist & Rate Limit Check (`AuthRedisCache`):**
   * `AuthService` depends directly on `AuthRedisCache` (Redis 7.2 cluster, Tier-0) to verify token revocation lists and active OTP session locks.
   * **Operational Constraint:** The default Redis connection pool in `AuthService` is capped at 50 connections unless overridden via config map. High-concurrency token refresh spikes can saturate this pool and trigger `504 Gateway Timeout` errors.

2. **SMS OTP Dispatch (`TwilioGateway`):**
   * For all phone-based authentication flows, `AuthService` delegates SMS challenge generation and delivery to `TwilioGateway` (Node.js service, Tier-1), maintained by Rohan Gupta.
   * If `TwilioGateway` is unreachable or rate-limited, `AuthService` fails closed and blocks account creation.

3. **Identity Risk Scoring (`FraudCheckService`):**
   * Before issuing a final session token, `AuthService` calls `FraudCheckService` (Python, Tier-1) with the user's verified phone number and device fingerprint.

## 3. Failure Modes & Escalation
* **Tier-0 Outages:** Any failure in `AuthService` or `AuthRedisCache` halts all user logins and checkouts. Page Priya Sharma immediately.
* **SMS Delivery Failures:** Escalate `TwilioGateway` errors to Rohan Gupta.