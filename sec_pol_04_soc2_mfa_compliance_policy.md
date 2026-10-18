# SEC-POL-04: Mandatory Multi-Factor Authentication & Fraud Governance

* **Document ID:** `doc_soc2_p4`
* **Document Type:** Security & Compliance Policy (SOC2 Type II)
* **Policy Owner:** Marcus Vance (`marcus@company.internal`, Head of Security & Compliance)
* **Governed Entities:** `PhoneSignup`, `OneClickCheckout`, `FraudCheckService`, `PaymentGateway`
* **Enforcement Level:** Mandatory (Blocker for Production Releases)

---

## 1. Policy Mandate
To maintain SOC2 Type II and PCI-DSS compliance, all user accounts capable of initiating financial transactions through `PaymentGateway` or `OneClickCheckout` must undergo verified Multi-Factor Authentication (MFA) at signup.

## 2. Phone Verification Requirement (Section 4.1)
1. **Mandatory Phone Binding:** The `PhoneSignup` feature must collect and verify an E.164 phone number via SMS OTP (`TwilioGateway`) prior to account activation.
2. **Downstream Dependency in Fraud Profiling:** `FraudCheckService` requires a verified phone carrier record to compute a user's initial risk score.
3. **Impact of Removing Phone Numbers:**
   * Making phone numbers optional on `PhoneSignup` without an approved fallback causes `FraudCheckService` to reject the payload with `HTTP 422 Unprocessable Entity`.
   * Because `PaymentGateway` depends on `FraudCheckService` clearance, any user registered without a verified phone factor will be automatically blocked from using `OneClickCheckout`.

## 3. Exception & Waiver Protocol
Engineering and Product teams (`Anita Roy`, `Priya Sharma`) **cannot** disable SMS 2FA or make phone numbers optional in `PhoneSignup` without explicit written sign-off from **Marcus Vance** and an active email-2FA fallback rule committed to the system graph.