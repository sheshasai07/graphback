# PRD-2026: Seamless User Onboarding & PhoneSignup Specification

* **Document ID:** `doc_prd_signup`
* **Document Type:** Product Requirement Document (PRD)
* **Product Owner:** Anita Roy (`anita@company.internal`, Product Manager - Growth)
* **Engineering Lead:** Priya Sharma (`priya@company.internal`)
* **Specified Feature:** `PhoneSignup`

---

## 1. Product Goal & Current Architecture
The `PhoneSignup` feature is our primary user acquisition funnel in Production. Currently, `PhoneSignup` requires every new user to enter a mobile number and verify a 6-digit SMS OTP before reaching the dashboard.

### Technical Touchpoints:
* **Frontend Feature:** `PhoneSignup`
* **Primary Backend Call:** Invokes `AuthService` (`/auth/v1/signup`), which triggers `TwilioGateway` for SMS delivery and `FraudCheckService` for risk validation.
* **Governance:** Bound by `SEC-POL-04` (owned by Marcus Vance).

## 2. Proposed Q4 Change Request: "Optional Phone Number at Signup"
* **Proposal:** Growth team (Anita Roy) wants to make the phone number field optional during initial signup to reduce user drop-off caused by international SMS delays.
* **Known Blast Radius & Blockers:**
  1. `AuthService` schema currently enforces `phone_number NOT NULL`.
  2. `FraudCheckService` will fail risk checks for users missing phone metadata, breaking `PaymentGateway` and `OneClickCheckout` downstream.
  3. Requires formal compliance waiver from **Marcus Vance** (`SEC-POL-04`) to allow Email-only 2FA for free-tier accounts.