# MASTER DEVELOPMENT DIRECTIVE — SWEEP ENTERPRISE INTELLIGENCE

**Project:** SWEEP

**Development agent:** Astra

**Primary goal:** Transform SWEEP into a commercially viable, modular AI-powered enterprise operations platform without compromising its existing companion functionality.

**Execution priority:** Functional software, demonstrable customer value, data accuracy, security, maintainability and measurable performance.

You are acting as SWEEP's principal software architect, senior full-stack engineer, AI systems engineer, data engineer, product strategist, QA engineer and technical documentation lead.

You must investigate the existing project before making changes. This is an implementation assignment, not merely a request to describe what might be built.

---

# PART 1 — MANDATORY EXISTING-PROJECT AUDIT

Before implementing anything:

1. Inspect the entire accessible SWEEP repository and identify its existing architecture, tech stack, models, agents, integrations, UI, storage, APIs and deployment method.
2. Identify working features, unfinished features, duplicated systems, technical debt, security problems and missing tests.
3. Preserve working functionality and existing user data.
4. Reuse or extend existing infrastructure instead of introducing another incompatible framework.
5. Document every proposed architectural change.
6. Identify which changes can be made safely and which require migration or approval.
7. Do not falsely report code, tests, datasets or features as implemented.
8. Establish the baseline performance, available hardware and existing operating costs.
9. Detect existing SWEEP features such as local execution, cloud models, plugin integrations, computer control, voice interaction and optional distributed resource sharing.
10. Maintain compatibility wherever technically possible.

**Deliverable:** `docs/ENTERPRISE_ARCHITECTURE_AUDIT.md`

Include a dependency map, existing module inventory, implementation risks and a prioritized development plan.

# PART 2 — CORE PRODUCT STRATEGY

SWEEP will remain a general-purpose AI companion with optional specialized enterprise capabilities.

Do not remove its original functionality.

Create a shared enterprise framework with four modules:

- **SWEEP TradeOps:** AI-assisted trade-document analysis, discrepancy detection and workflow automation.
- **SWEEP Vision:** Visual quality inspection for selected manufacturing applications.
- **SWEEP TestLab:** Synthetic business-document creation and AI-workflow testing.
- **SWEEP Evidence:** Authorized incident-evidence analysis and forensic triage.

Prioritize TradeOps. The other modules must remain independently installable or enableable, with shared foundations but separate business logic.

Do not claim all four modules are finished if they are not.

# PART 3 — TRADEOPS: FIRST COMMERCIAL PRODUCT

Build the first functional version around small exporters, importers, customs-service providers and freight brokers that use email, PDFs, spreadsheets and manually operated web portals.

## 3.1 Document ingestion

Support, where technically feasible:

- Commercial invoices
- Packing lists
- Purchase orders
- Bills of lading
- Air waybills
- Shipping instructions
- Supplier correspondence
- Product catalogs
- Certificates of origin
- Customs-related documents
- Existing Excel/CSV records

Create file ingestion with format detection, text extraction, OCR, table parsing, metadata extraction and structured document representation.

Retain links between extracted data and original document pages or regions.

Never fabricate missing document fields.

## 3.2 Intelligent data extraction

Extract and normalize:

- Buyer and seller
- Addresses and countries
- Document numbers and dates
- Shipment identifiers
- Products and descriptions
- Quantities and measurement units
- Unit prices and totals
- Currencies
- Freight and insurance values where available
- Origin and destination countries
- HS codes when explicitly supplied
- Incoterms when explicitly supplied
- Container and tracking references

Store every extracted field with:

- Original source document
- Page or region reference
- Extraction method
- Confidence indicator or uncertainty status
- Validation result
- Human correction history

A model-generated confidence score must not be presented as calibrated probability unless calibration has been validated experimentally.

## 3.3 Document reconciliation

Compare related documents automatically.

Detect:

- Mismatched quantities
- Inconsistent product descriptions
- Different currencies
- Incorrect totals
- Conflicting addresses
- Missing required fields
- Duplicate records
- Unexpected changes between document versions
- Inconsistent dates
- Unmatched shipping references

Use deterministic calculations and validation rules whenever possible.

Present discrepancies with references to their actual sources.

## 3.4 AI workflow assistant

Allow commands such as:

"Compare these invoices against the packing lists."

"Find all shipments with inconsistent quantities."

"Prepare an exception report for this shipment."

"Organize the supplier documents for this order."

"Create a draft email requesting corrected documents."

"Update the shipment spreadsheet after I approve the changes."

"Show me which records still require manual review."

Each command must execute a real workflow, not return simulated results.

## 3.5 Trade intelligence

Investigate integration with:

- UN Comtrade API
- EU Access2Markets
- Official customs and trade classification resources
- India DGFT resources
- Other official country-specific trade datasets

Check API access requirements, permitted reuse, licensing, rate limits and geographic coverage.

Distinguish historical trade statistics from live customs or shipment information.

Where official data is unavailable, return an explicit limitation rather than generating unverified tariff rates, regulatory requirements or classifications.

Treat HS-code suggestions and customs guidance as reviewable assistance, not authoritative legal determinations.

# PART 4 — SWEEP VISION

Create a modular computer-vision capability for visual quality inspection.

Potential tasks:

- Detect missing components
- Identify packaging damage
- Compare objects against reference images
- Detect visible manufacturing defects
- Identify anomalous product appearances
- Generate timestamped inspection reports

Research and evaluate appropriate open datasets and models, including:

- MVTec AD
- VisA industrial anomaly detection
- Roboflow Universe datasets
- Ultralytics-based object detection where licensing permits
- OpenCV-based deterministic image analysis

Before use, verify dataset licenses, redistribution permissions, commercial rights and suitability for the actual inspection task.

Do not imply that benchmark performance transfers directly to a customer's factory.

Require task-specific validation with representative production images.

Use this module initially as a prototype until sufficient manufacturing data is available.

# PART 5 — SWEEP TESTLAB

Create an optional synthetic-data and software-testing module.

Functions:

- Generate fictional business invoices
- Produce packing lists and purchase orders
- Create deliberately inconsistent document sets
- Generate OCR stress-test files
- Produce multilingual test documents
- Simulate missing fields and corrupted records
- Create reproducible agent workflow tests
- Evaluate document extraction and reconciliation accuracy

All generated business records must be clearly identified as synthetic.

Avoid reproducing real personal or proprietary information.

Provide deterministic seeds, expected outputs and repeatable benchmark runs.

This module should serve SWEEP's internal quality assurance before being considered a separate commercial product.

# PART 6 — SWEEP EVIDENCE

Build an authorized investigation workspace focused on enterprise incident triage.

Functions:

- Ingest permitted log files
- Organize digital evidence
- Extract timestamps and metadata
- Calculate cryptographic file hashes
- Correlate events across files
- Generate investigation timelines
- Search evidence collections
- Produce source-linked findings
- Preserve records of changes and analysis operations

Preserve original evidence and use read-only analysis where possible.

Do not modify source files without explicit approval.

Support audit trails and evidence provenance.

Do not falsely claim forensic certification, admissibility, or validated chain of custody.

Keep this module separate from TradeOps. It is not required for the first commercial release.

# PART 7 — SHARED AGENT FRAMEWORK

Create or extend a central agent orchestration system supporting:

- Task decomposition
- Tool discovery
- Tool selection
- Model routing
- Long-running job state
- Retry handling
- Failure recovery
- Human approval steps
- Parallel processing when appropriate
- Execution history
- Cost tracking
- Result verification

Use appropriate open-source orchestration systems only if they improve the existing architecture.

Evaluate LangGraph and comparable frameworks against SWEEP's existing implementation before choosing one.

Every workflow should have clear input schemas, output schemas, permissions and execution logs.

# PART 8 — LOCAL, CLOUD AND OPTIONAL SHARED COMPUTING

Maintain support for local and cloud execution where practical.

Use local processing for inexpensive or privacy-sensitive tasks where hardware permits.

Use cloud processing for workloads that exceed local capabilities, subject to user permissions and cost settings.

SWEEP's previously planned platform resource-sharing system is optional and opt-in.

The entire application must remain functional when no other users contribute computational resources.

Do not make enterprise processing dependent on distributed community compute.

When community computing is enabled, never expose another customer's confidential documents, credentials or raw sensitive business data to untrusted peers.

Use only workloads suitable for the trust boundary and explicit permissions.

# PART 9 — SECURITY AND ENTERPRISE CONTROLS

Implement security appropriate to the project's maturity.

Requirements:

- Authentication and authorization
- Workspace separation
- Tenant-isolated storage and retrieval
- Encryption in transit
- Secure secret storage
- Least-privilege connectors
- Explicit user authorization for external actions
- Audit logs
- Configurable data retention
- Export and deletion controls
- File validation and malware scanning
- Prompt-injection defenses
- SSRF protections
- Sandboxed execution where applicable
- API rate limiting
- Usage quotas and cost controls

Treat uploaded documents, webpages and retrieved data as untrusted input.

External content must never be able to rewrite system instructions or authorize an action.

Require explicit user confirmation before sending external messages, submitting documents, changing business records or performing irreversible operations.

Do not claim SOC 2, ISO 27001, GDPR compliance or similar certification unless independently established.

# PART 10 — USER EXPERIENCE

Maintain SWEEP's identity as an elegant, highly interactive desktop companion.

Design inspiration:

- Apple: clarity and restrained visual design
- Claude: readable conversations and productive workspaces
- Modern professional desktop applications: flexible panels and contextual controls

Avoid unnecessarily flashy graphics and heavy animations.

Build interfaces for:

- Conversation
- File uploads
- Project workspaces
- Document comparison
- Evidence/source inspection
- Workflow automation
- Human approvals
- Errors and warnings
- Performance monitoring
- Billing/usage visibility
- Reports
- Plugin management

For TradeOps, support an intuitive workspace where two or more documents can be compared side by side, discrepancies are highlighted, and clicking a result navigates to the source evidence.

All controls must operate.

Do not implement decorative buttons that appear functional but have no real behavior.

# PART 11 — DATASETS AND EVALUATION

Research and evaluate:

Document datasets:
- Thoughtworks Document Processing Benchmark
- FUNSD
- CORD
- WildReceipt
- DocVQA
- DocILE
- CUAD
- Suitable commercial-permitted invoice datasets

Vision datasets:
- MVTec AD
- VisA
- Roboflow Universe selections

Public trade information:
- UN Comtrade
- EU Access2Markets
- DGFT and other official national trade resources

For every source, document:

1. Official URL
2. Dataset owner
3. Data type
4. Size and coverage
5. License
6. Commercial-use permission
7. Data quality
8. Geographic relevance
9. Update frequency
10. Known limitations
11. Whether downloaded or integrated
12. Evaluation suitability

Do not download everything indiscriminately.

Never mix noncommercial or research-only data into commercial model training without appropriate rights.

Begin with reproducible evaluation benchmarks.

Use real customer data only with appropriate authorization.

# PART 12 — QUALITY TARGETS

Design measurable quality gates.

Proposed initial targets, to be validated and adjusted using actual representative datasets:

- At least 95% exact-match accuracy for a clearly specified set of critical fields on the selected pilot document types
- At least 98% precision for high-severity discrepancy alerts
- Zero known cross-tenant data exposures in security testing
- Reliable handling of partial document uploads and processing failures
- Complete source attribution for material extracted fields
- Explicit uncertainty indicators for ambiguous outputs
- Documented latency and processing cost per document

These are engineering targets, not promises of achievable performance.

Report actual measured results and sample sizes. Never substitute LLM-generated results for evaluation.

# PART 13 — MONETIZATION AND BUSINESS EXPERIMENTS

Create the foundations for commercial plans.

Possible plans:

- Free evaluation
- Individual professional
- Small business
- Team
- Enterprise/custom deployment

Potential billing dimensions:

- Number of users
- Documents processed
- AI compute usage
- Automation runs
- Storage
- Premium integrations
- Dedicated hosting

Do not permanently hardcode prices before customer validation.

Implement metering and design pricing experiments around measured infrastructure costs and customer willingness to pay.

Track:

- Customer activation
- Time to first completed workflow
- Document processing accuracy
- Time saved
- Paid conversion
- Retention
- Cost per workflow
- Gross margin
- Customer support burden

# PART 14 — IMPLEMENTATION ROADMAP

## Phase 0: Audit and validation

- Inspect SWEEP
- Identify dependencies and gaps
- Verify resources
- Define baseline metrics
- Interview prospective customers if access is available
- Select one precise initial trade workflow

## Phase 1: Shared foundation

- File ingestion
- OCR and parsing
- Task orchestration
- Workspace management
- Authorization
- Logging
- Source attribution
- Tests

## Phase 2: TradeOps MVP

- Commercial invoice extraction
- Packing-list extraction
- Cross-document comparison
- Discrepancy highlighting
- CSV/Excel export
- Human correction
- Draft communications
- Controlled workflow automation

## Phase 3: Pilot readiness

- Run benchmark tests
- Improve extraction accuracy
- Add cost monitoring
- Test security boundaries
- Create customer onboarding
- Test deployment and recovery
- Prepare pilot reporting

## Phase 4: Commercial readiness

- Billing infrastructure
- Production monitoring
- Team permissions
- Customer support tools
- Reliable updates
- Performance optimization
- Documentation

## Phase 5: Additional modules

Build SWEEP Vision, TestLab and Evidence incrementally, based on technical feasibility and demonstrated customer demand.

Do not let later modules delay the first paid pilot.

# PART 15 — REQUIRED PROJECT DELIVERABLES

Create or update:

- `docs/ENTERPRISE_ARCHITECTURE_AUDIT.md`
- `docs/IMPLEMENTATION_ROADMAP.md`
- `docs/SOURCE_AND_DATASET_REGISTRY.md`
- `docs/TRADEOPS_PRODUCT_SPEC.md`
- `docs/SECURITY_THREAT_MODEL.md`
- `docs/EVALUATION_PROTOCOL.md`
- `docs/MARKET_VALIDATION.md`
- `docs/COST_AND_UNIT_ECONOMICS.md`
- `docs/DEPLOYMENT_GUIDE.md`

Include actual implemented source code, database migrations, tests, configuration templates and reproducible setup instructions.

Maintain a feature-status matrix marking each capability:

- Implemented and tested
- Implemented but unverified
- Partially implemented
- Planned
- Blocked

Do not report placeholder interfaces as completed features.

# FINAL EXECUTION INSTRUCTION

Start by investigating the existing SWEEP codebase and producing the architecture audit.

Then implement Phase 1 and the initial TradeOps MVP in logically separated, testable increments, using the existing project structure whenever appropriate.

Run tests and correct identified failures. Where access, credentials, external services or datasets are missing, implement safe interfaces or test doubles, document the blocker precisely and continue with independent tasks.

Do not invent test results, source availability, enterprise readiness or commercial demand.

Prefer reliable, cohesive functionality to unnecessary complexity.

**Success criterion:** SWEEP can receive actual business documents, extract source-linked structured fields, compare them, identify genuine discrepancies, generate a reviewable report and complete an authorized follow-up workflow, with measured performance and reproducible tests.

Build toward a sustainable, commercially useful product—not an impressive but unreliable demonstration.