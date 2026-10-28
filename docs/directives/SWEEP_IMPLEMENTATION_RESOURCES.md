# PART 11 — VERIFIED DATASETS, APIs, OPEN-SOURCE TOOLS AND IMPLEMENTATION RESOURCES

Astra, use the following resources to research, develop, test, and improve SWEEP's enterprise AI capabilities.

**Important instructions:**
- Visit and verify each source before using it.
- Check repository maintenance, access requirements, version compatibility, and licensing.
- Do not assume every dataset permits commercial use.
- Download and integrate only resources relevant to the modules being implemented.
- Prefer official sources and repositories.
- If a URL is unavailable, locate its official replacement and document the change.
- Never fabricate downloaded datasets, API access, test results, or completed integrations.

## A. Document Intelligence and OCR

These resources are intended primarily for SWEEP TradeOps.

**1. Docling — Document processing**

https://github.com/docling-project/docling

Purpose: Extract structured information from PDFs and other business documents.

**2. PaddleOCR — Multilingual OCR**

https://github.com/PaddlePaddle/PaddleOCR

Purpose: Extract text from scanned invoices, shipping documents, images, and multilingual records.

**3. Thoughtworks Document Processing Benchmark**

https://huggingface.co/datasets/thoughtworks/document-processing-benchmark

Purpose: Evaluate document-understanding systems across different business-document formats.

Important: Review the license of each constituent dataset.

**4. FUNSD — Form understanding**

https://guillaumejaume.github.io/FUNSD/

Purpose: Evaluate form text recognition and field relationships.

**5. CORD — Receipt understanding**

https://github.com/clovaai/cord

Purpose: Evaluate structured extraction from receipts.

**6. WildReceipt**

https://github.com/clovaai/wildreceipt

Purpose: Evaluate receipt information extraction and entity recognition.

**7. DocVQA**

https://www.docvqa.org/

Purpose: Benchmark document visual question answering.

**8. DocILE**

https://github.com/rossumai/docile

Purpose: Evaluate key information extraction and line-item recognition from business documents.

**9. CUAD — Contract Understanding Atticus Dataset**

https://www.atticusprojectai.org/cuad

Purpose: Research contract clause identification and document analysis.

**10. Invoice and Receipt Dataset Collection**

https://huggingface.co/collections/drew-ipp/invoice-and-receipt-extraction-datasets

Purpose: Discover additional invoice and receipt datasets for extraction evaluation.

**11. Business Documents Repository**

https://github.com/upbrains-ai/Business-Documents-Repository-for-AI

Purpose: Discover business-document datasets and related research resources.

For every dataset, verify usage rights before commercial training, fine-tuning, redistribution, or hosted evaluation.

## B. International Trade Data and Official Sources

**12. UN Comtrade API**

https://uncomtrade.org/docs/un-comtrade-api/

Purpose: Obtain international trade statistics for analytical workflows.

**13. UN Comtrade Database**

https://comtradeplus.un.org/

Purpose: Explore commodity trade patterns and historical international trade activity.

**14. EU Access2Markets**

https://trade.ec.europa.eu/access-to-markets/en/home

Purpose: Research tariffs, trade requirements, rules of origin, and cross-border trade conditions.

**15. Directorate General of Foreign Trade — India**

https://www.dgft.gov.in/

Purpose: Access official Indian foreign-trade information, notices and related resources.

**16. World Bank Open Data**

https://data.worldbank.org/

Purpose: Obtain economic and international trade indicators for research dashboards.

**17. WTO Data Portal**

https://data.wto.org/

Purpose: Research international trade statistics and related indicators.

**18. UN/CEFACT**

https://unece.org/trade/uncefact

Purpose: Investigate international trade-data standards and document interoperability.

**19. World Customs Organization**

https://www.wcoomd.org/

Purpose: Research customs classification standards and official guidance.

Important: Some sources provide websites or reference information rather than unrestricted APIs. Do not scrape restricted portals or invent API endpoints.

## C. Computer Vision and Industrial Inspection

These resources support SWEEP Vision.

**20. MVTec Anomaly Detection Dataset**

https://www.mvtec.com/research-teaching/datasets/mvtec-ad

Purpose: Industrial anomaly detection and defect-localization research.

Important: Dataset usage is restricted; obtain appropriate rights before commercial deployment or training.

**21. VisA Industrial Anomaly Dataset**

https://registry.opendata.aws/visa/

Purpose: Evaluate visual anomaly detection using industrial product images.

**22. Roboflow Universe**

https://universe.roboflow.com/

Purpose: Discover computer-vision datasets for different inspection tasks.

Licensing must be checked individually.

**23. OpenCV**

https://github.com/opencv/opencv

Purpose: Image processing, measurement, image comparison and computer-vision pipelines.

**24. Ultralytics**

https://github.com/ultralytics/ultralytics

Purpose: Evaluate object detection, segmentation and related vision tasks.

Review AGPL and commercial licensing requirements before integration.

## D. Agent Orchestration and Automation

**25. LangGraph**

https://github.com/langchain-ai/langgraph

Purpose: Stateful agent orchestration and multi-step workflows.

**26. Microsoft Playwright**

https://github.com/microsoft/playwright

Purpose: Browser interaction, workflow automation and automated testing.

**27. Model Context Protocol**

https://modelcontextprotocol.io/

Purpose: Investigate interoperable tool and data-source connections.

**28. Apache Airflow**

https://airflow.apache.org/

Purpose: Evaluate scheduled data processing and workflow orchestration where appropriate.

**29. Temporal**

https://temporal.io/

Purpose: Evaluate durable, fault-tolerant business workflow execution.

Do not integrate every framework. Choose the smallest maintainable set appropriate for SWEEP's current architecture.

## E. Security and Enterprise AI Reliability

**30. OWASP Generative AI Security Project**

https://genai.owasp.org/

Purpose: Protect AI systems from prompt injection, unsafe tool use, data leakage and related risks.

**31. MITRE ATLAS**

https://atlas.mitre.org/

Purpose: Research adversarial threats involving AI systems.

**32. NIST AI Risk Management Framework**

https://www.nist.gov/itl/ai-risk-management-framework

Purpose: Structure trustworthy AI design and evaluation.

**33. OWASP Application Security Verification Standard**

https://owasp.org/www-project-application-security-verification-standard/

Purpose: Guide application security testing and control design.

## F. Synthetic Data and AI Evaluation

**34. Microsoft Presidio**

https://github.com/microsoft/presidio

Purpose: Identify and help anonymize sensitive information when preparing authorized test data.

**35. SDV — Synthetic Data Vault**

https://github.com/sdv-dev/SDV

Purpose: Evaluate synthetic structured-data generation.

**36. Ragas**

https://github.com/explodinggradients/ragas

Purpose: Evaluate selected LLM and retrieval-augmented workflows.

**37. DeepEval**

https://github.com/confident-ai/deepeval

Purpose: Evaluate AI outputs and agent workflows with reproducible test cases.

**38. Evidently AI**

https://github.com/evidentlyai/evidently

Purpose: Monitor data quality, model behavior and evaluation metrics.

## G. Digital Forensics and Incident Analysis

These resources support the optional SWEEP Evidence module.

**39. NIST Computer Forensics Tool Testing**

https://www.cftt.nist.gov/

Purpose: Research testing methodology and validation requirements for forensic tools.

**40. Digital Corpora**

https://digitalcorpora.org/

Purpose: Discover forensic research corpora and sample evidence datasets.

Check individual dataset access rights and restrictions.

**41. Volatility 3**

https://github.com/volatilityfoundation/volatility3

Purpose: Research memory-forensics analysis for authorized investigations.

**42. Timesketch**

https://github.com/google/timesketch

Purpose: Investigate collaborative event-timeline analysis and incident-response workflows.

**43. MITRE ATT&CK**

https://attack.mitre.org/

Purpose: Research known adversarial techniques for defensive incident analysis.

---

# REQUIRED IMPLEMENTATION PROCEDURE

For every resource above:

1. Verify the URL and source authenticity.
2. Record current availability and maintenance status.
3. Identify license and commercial-use restrictions.
4. Assess how the resource supports SWEEP.
5. Determine whether integration is technically justified.
6. Record installation instructions and dependencies.
7. Evaluate actual performance using permitted sample data.
8. Record computational and infrastructure requirements.
9. Document security and privacy risks.
10. Reject unsuitable resources and explain why.

Create:

`docs/SOURCE_AND_DATASET_REGISTRY.md`

Include a table containing:

- Resource name
- Official URL
- Dataset or software category
- Relevant SWEEP module
- Access method
- License
- Commercial-use status
- Size and data coverage
- Update and maintenance status
- Integration status
- Benchmark results, where measured
- Known limitations

**Priority order:**

First, implement document processing using suitable OCR and parsing systems.

Second, build business-document extraction, reconciliation, and source-linked reporting.

Third, connect permitted trade-information sources.

Fourth, establish agent execution, testing, security and cost monitoring.

Fifth, evaluate visual inspection, synthetic-data generation and authorized digital-forensics modules.

**Final instruction:** These resources are a research and development starting point, not a requirement to install all 43. Investigate them systematically, use only what improves SWEEP, and prioritize creating a secure, reliable and commercially valuable first enterprise product.