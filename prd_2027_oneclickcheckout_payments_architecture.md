# PRD-2027: OneClickCheckout & PaymentGateway Integration

* **Document ID:** `doc_prd_checkout`
* **Document Type:** Product & Technical Specification
* **Engineering Maintainer:** Priya Sharma (`priya@company.internal`, Staff Backend Engineer)
* **Security Approver:** Marcus Vance (`marcus@company.internal`, Head of Security & Compliance)
* **Target Entities:** `OneClickCheckout`, `PaymentGateway`, `FraudCheckService`

---

## 1. Feature Summary
`OneClickCheckout` (currently in Beta) enables returning users to complete purchases instantly without re-entering billing details.

## 2. Service Dependency Chain
1. `OneClickCheckout` calls `PaymentGateway` (Java service, Tier-0 criticality, maintained by Priya Sharma).
2. `PaymentGateway` has a hard runtime dependency (`DEPENDS_ON`) on `FraudCheckService` (Python service, Tier-1, governed by Marcus Vance).
3. Before capturing funds, `PaymentGateway` queries `FraudCheckService` to confirm that the user's session token from `AuthService` includes an active MFA verification flag (`sms_verified: true` or `compliance_override: email_2fa`).

## 3. Operational Impact
If `AuthService` or `PhoneSignup` stops passing verified phone metadata—or if `FraudCheckService` goes down—`PaymentGateway` automatically declines all `OneClickCheckout` transactions to prevent unverified chargebacks.