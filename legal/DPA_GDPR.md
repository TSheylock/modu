
# Data Processing Agreement (DPA)
**Version:** 1.0
**Effective Date:** 2025-12-05

This Data Processing Agreement ("DPA") is entered into between **SASOK Project** ("Provider") and the undersigned customer ("Customer") and is incorporated into the Master Services Agreement ("MSA") between the parties. This DPA will be effective from the date it is counter-signed by the Provider.

## 1. Definitions
*   **"GDPR"** means Regulation (EU) 2016/679 of the European Parliament and of the Council of 27 April 2016.
*   **"Personal Data"**, **"Data Subject"**, **"Processing"**, **"Controller"**, and **"Processor"** shall have the meanings given to them in the GDPR.
*   **"Customer Data"** means all Personal Data that Customer provides to the Provider for processing in connection with the Services.

## 2. Roles and Responsibilities
*   **Controller:** The Customer is the Controller of the Customer Data.
*   **Processor:** The Provider is the Processor of the Customer Data. The Provider will process Customer Data only on behalf of the Customer and in accordance with its documented instructions.

## 3. Scope of Processing
*   **Subject Matter:** The processing of Customer Data in connection with the provision of the SASOK Platform services as described in the MSA.
*   **Duration:** For the term of the MSA.
*   **Nature and Purpose:** To enable the Customer to use the Platform's features for symbiotic cognitive-emotional analysis, including real-time data capture, emotional state vectorization, pattern analysis, and data storage.
*   **Categories of Data Subjects:** End-users authorized by the Customer to use the Platform.
*   **Types of Personal Data:**
    *   **Directly Provided:** User profile information (name, email).
    *   **Collected via Platform:** Real-time visual, audial, biometric, and behavioral data streams, and the emotional/cognitive metadata derived from them. This is considered **Special Category Data** under GDPR Article 9.

## 4. Obligations of the Processor (Provider)
The Provider shall:
a) Process Customer Data only in accordance with the Customer's lawful and documented instructions.
b) Ensure that all personnel authorized to process Customer Data are bound by a duty of confidentiality.
c) Implement and maintain appropriate technical and organizational security measures to protect Customer Data against unauthorized or unlawful processing and against accidental loss, destruction, or damage. These measures are detailed in Appendix 1.
d) Not engage any sub-processor without the prior specific or general written authorization of the Customer. A list of current sub-processors is provided in Appendix 2.
e) Assist the Customer, by appropriate technical and organizational measures, in fulfilling its obligation to respond to requests from Data Subjects exercising their rights under the GDPR.
f) Notify the Customer without undue delay after becoming aware of a Personal Data Breach.
g) Upon termination of the MSA, delete or return all Customer Data to the Customer, at the Customer's choice.
h) Make available to the Customer all information necessary to demonstrate compliance with the obligations laid down in this DPA and allow for and contribute to audits.

## 5. Obligations of the Controller (Customer)
The Customer represents and warrants that:
a) It has a lawful basis for the processing of all Customer Data provided to the Provider.
b) It has obtained all necessary and explicit consents from Data Subjects for the processing of their Personal Data, especially for Special Category Data (biometrics, health-inferred data).

## 6. International Data Transfers
The Provider shall not transfer Customer Data outside the European Economic Area (EEA) without ensuring that the transfer is compliant with the requirements of the GDPR, typically through the use of Standard Contractual Clauses (SCCs).

## 7. Liability and Indemnity
The liability of each party under this DPA shall be subject to the limitations and exclusions of liability set out in the MSA.

---
## Appendix 1: Technical and Organizational Security Measures
*   **Encryption:** All data is encrypted in transit (TLS 1.3) and at rest (AES-256).
*   **Access Control:** Strict role-based access control (RBAC) and the principle of least privilege are enforced.
*   **Pseudonymization:** Emotional vectors and biometric data are pseudonymized by default, separating them from direct personal identifiers.
*   **Logging and Monitoring:** Comprehensive logging of access and actions is in place to detect and respond to security incidents.
*   **Data Minimization:** The Platform is designed to collect only the data strictly necessary for its functioning. Raw data streams are processed transiently and not stored by default.

## Appendix 2: List of Sub-processors
| Service Provider | Purpose | Location (Data Center) |
|------------------|---------|------------------------|
| Amazon Web Services (AWS) | Cloud Infrastructure (Compute, Storage) | Frankfurt, Germany (eu-central-1) |
| MongoDB, Inc. | Managed Database Hosting (MongoDB Atlas) | Frankfurt, Germany (eu-central-1) |
| Neo4j, Inc. | Managed Graph Database Hosting (Neo4j Aura) | Frankfurt, Germany (eu-central-1) |
| Redis Labs | Managed In-Memory Cache | Frankfurt, Germany (eu-central-1) |
---

**IN WITNESS WHEREOF,** the parties have caused this DPA to be executed by their duly authorized representatives.

**Customer:**
Signature: _________________________
Name: _________________________
Date: _________________________

**Provider (SASOK Project):**
Signature: _________________________
Name: Timmy Sheylock
Date: 2025-12-05
