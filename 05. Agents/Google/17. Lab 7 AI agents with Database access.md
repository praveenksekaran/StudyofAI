Build a Foundational AI Agent With Secure Database Access
=========================================================


Overview
--------

For security and stability, AI agents should not have direct access to production databases. Instead, a secure API should act as a controlled intermediary between the agent and the data.

**The challenge**: Enterprise AI agents often need to query a mix of data types simultaneously from the same database, such as structured records and unstructured documents. This creates a challenge of how to provide unified access without exposing the database to the risks of direct AI interaction.

**The solution**: Create a secure API gateway. This separates the AI from the data layer, allowing the agent to request information in a controlled way without ever touching the live database directly.

In Google Cloud, you can implement this solution by building a production-grade agent using this three-tier architecture:

*   1\. **Data Layer:** An **AlloyDB** database configured for high-speed vector search.
*   2\. **Secure API Layer:** The **MCP Toolbox for Databases**, deployed on Cloud Run as a private, secure API endpoint that exposes specific data access "tools."
*   3\. **Agent Layer:** An intelligent agent built with the **Google Cloud Agent Development Kit (ADK)** that securely authenticates and consumes the API layer to answer questions.

This pattern ensures your agent is powerful yet secure, only accessing data through approved, audited channels. Your final agent will securely authenticate and consume the API, empowering an insurance adjuster to query policy details and find repair articles using natural language—all without ever touching the production database directly.

### Enterprise use cases

*   • **Finance**: An agent for a financial advisor queries a client's structured portfolio holdings and unstructured market analysis reports to answer questions about investment performance.
*   • **Retail**: A customer service agent accesses a customer's structured order history and unstructured product troubleshooting guides to resolve an issue.
*   • **Logistics**: A supply chain manager's agent pulls structured inventory data and unstructured shipping lane risk assessments to find the most efficient delivery route.

### Objectives

In this lab, you learn how to:

*   Configure an AlloyDB database for vector search and store embeddings.
*   Create a simple API wrapper for MCP Toolbox for Databases using FastAPI.
*   Deploy the MCP Toolbox as a private, secure API service on Cloud Run.
*   Set up the Agent Development Kit (ADK) environment according to best practices.
*   Build a custom ADK tool that securely calls a private Cloud Run service.
*   Create and run an ADK agent that uses the custom tool to answer questions.

Setup and requirements
----------------------

### Activate Cloud Shell

Cloud Shell is a virtual machine that contains development tools. It offers a persistent 5-GB home directory and runs on Google Cloud. Cloud Shell provides command-line access to your Google Cloud resources. `gcloud` is the command-line tool for Google Cloud. It comes pre-installed on Cloud Shell and supports tab completion.

*   Click the **Activate Cloud Shell** button (Activate Cloud Shell icon) at the top right of the console.
*   Click **Continue**.

  ```
  # List the active account name:
  gcloud auth list
  
  # List the project ID:
  gcloud config list project
  ```

Part 1: The backend data layer
------------------------------

Task 1. Configure an AlloyDB database to support vector search
--------------------------------------------------------------

The provisioning of an AlloyDB cluster named `cymbal-cluster` and an instance named `cymbal-instance` began when you started the lab.

In this task, you enable key extensions in the default AlloyDB database named `postgres` and grant the necessary permissions to support the generation, storage, and querying of text embeddings for vector search. You begin by using [AlloyDB Studio](https://cloud.google.com/alloydb/docs/manage-data-using-studio) to connect to the AlloyDB database and enable the [`vector`](https://cloud.google.com/alloydb/docs/ai#store-index-query-vectors) extension (note that the other necessary extension named [`google_ml_integration`](https://cloud.google.com/alloydb/docs/ai#generate_embeddings_and_text_predictions) is already enabled to allow access to Vertex AI endpoints). Last, you use [Identity and Access Management (IAM)](https://cloud.google.com/security/products/iam) to grant the appropriate Vertex AI role to the AlloyDB service account used to access Vertex AI resources.

### Enable extensions and permissions in AlloyDB for vector search

*   In the Google Cloud console, click the **Navigation menu** (Navigation menu icon) > **View All Products**. Under **Databases**, click **AlloyDB**.

*   In the AlloyDB menu, click **Clusters** to examine the cluster's details.

    It may take a few minutes for both the cluster and instance to be fully provisioned.

    When you see a **Status** of **Ready** (green checkmark) for both `cymbal-cluster` and `cymbal-instance`, you can proceed to the next step.

*   Click on the instance named `cymbal-instance`.

*   In the AlloyDB menu under **Primary Cluster**, click **AlloyDB Studio**.

*   Provide the following details to sign in, and click **Authenticate**.

|   |   |
|---|---|
| Property | Value |
| Database | Select `postgres` |
| User | Select `postgres` |
| Password | `changeme` |

*   On the AlloyDB Studio page, click **New SQL editor tab** to open a new query window.

*   To enable the `vector` extension, copy and paste the following query in the query window, and click **Run**.
  ```
  CREATE EXTENSION IF NOT EXISTS vector;
  ```
  When the query has executed successfully, you see a message that says Statement executed successfully.

*   In the query editor window, click **Clear** (along the same menu bar as **Run**) to remove the previous query.

*  To grant the default database user named `postgres` the permission to execute the embedding function, copy and paste the following query and click **Run**.
  ```
  GRANT EXECUTE ON FUNCTION embedding TO postgres;
  ```
  When the query has executed successfully, you see a message that says Statement executed successfully.

### Grant Vertex AI User role to the AlloyDB service account

Now that the database has the appropriate extensions and permissions, you can complete the [database integration with Vertex AI](https://cloud.google.com/alloydb/docs/ai/configure-vertex-ai) by granting the AlloyDB service account the appropriate role to access Vertex AI resources.

1. In the Google Cloud console, on the **Navigation menu** (Navigation menu icon), select **IAM & Admin** > **IAM**.
2. Click **Grant access**.
3. For **New principals**, enter the AlloyDB service account ID: **service- @gcp-sa-alloydb.iam.gserviceaccount.com**
4. For **Select a role**, select **Vertex AI** > **Vertex AI User**, and click **Apply**.
5. Click **Save**.

Task 2. Create a new table and load customer data
-------------------------------------------------

In this task, you create a new table containing various columns from the source data, plus an extra column for vector embeddings. Last, you load a sample of fictional customer data for the Cymbal Insurance company, which includes written summaries (stored in the column named **abstracts**) of the customer records.

1. Return to AlloyDB Studio. If you closed the tab, follow steps 1-4 in Task 1 to reopen it.
2. In the query editor window, click **Clear**.
3. To create a new table named `customer_records_data`, copy and paste the following query, and click **Run**.
  ```
  CREATE TABLE customer_records_data (
      id VARCHAR(25),
      type VARCHAR(25),
      number VARCHAR(20),
      country VARCHAR(2),
      date VARCHAR(20),
      abstract VARCHAR(300000),
      title VARCHAR(100000),
      kind VARCHAR(6),
      num_claims BIGINT,
      filename VARCHAR(100),
      withdrawn BIGINT,
      abstract_embeddings vector(768)
  );
  ```

The last column of the table, **abstract_embeddings**, is of the type `vector`, which will store the vector values you create in the next task.

When the query has executed successfully, you'll see a message that says Statement executed successfully.

4. In the AlloyDB Studio query editor, click **Clear**.
5. To load data into the table, copy and paste the following query, and click **Run**.

  ```
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('POL-10326103','POLICY','CUST-84321','US','2023-06-18','Standard HO-3 homeowners policy. Coverage includes: Dwelling ($500,000), Other Structures ($50,000), Personal Property ($250,000), and Loss of Use ($100,000). Includes a $1,000 deductible for all perils except for a 2% deductible for named storms. Endorsements include water backup and sump pump overflow coverage up to $10,000.','Homeowners Insurance HO-3 Policy','ACTIVE',1,'policy_84321_ho3.pdf',0,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('ART-10326507','ARTICLE','PLUMB-001','US','2022-01-20','This article details the standard procedure for mitigating residential water damage. Steps include: 1. Shutting off the main water supply. 2. Identifying the source of the leak (e.g., burst pipe, appliance failure). 3. Removing standing water using pumps or wet vacuums. 4. Drying the affected area with dehumidifiers and fans to prevent mold growth. 5. Assessing structural damage to drywall and flooring.','Standard Procedure for Mitigating Water Damage','CURNT',12,'kb_water_damage_mitigation.html',0,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('POL-10328303','POLICY','CUST-19873','US','2022-11-01','Comprehensive auto insurance policy for a 2022 sedan. Coverage includes: Bodily Injury Liability ($100k/$300k), Property Damage Liability ($50k), Collision ($500 deductible), and Comprehensive ($250 deductible). Includes rental reimbursement and roadside assistance endorsements. No at-fault accidents on record.','Comprehensive Auto Policy','ACTIVE',0,'policy_19873_auto.pdf',0,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('ART-10328431','ARTICLE','ROOF-003','US','2021-07-15','This guide provides instructions for the emergency tarping of a storm-damaged roof to prevent further water intrusion. It covers safety precautions for working at height, methods for securing a heavy-duty tarp over damaged sections using wood planks and nails, and how to create a watertight seal. This is a temporary fix pending professional repair.','Emergency Roof Tarping After Storm Damage','CURNT',35,'kb_roof_tarping.html',0,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('POL-10328505','POLICY','CUST-55429','CA','2024-01-05','Commercial property insurance for a small retail business. Policy covers the building, business personal property, and business income. Exclusions apply for flood and earthquake damage. Special endorsement for spoilage of perishable goods up to $25,000 due to equipment failure.','Commercial Property Insurance Policy','ACTIVE',0,'policy_55429_comm.pdf',0,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('ART-10328822','ARTICLE','AUTO-005','US','2023-03-10','An overview of assessing front-end collision damage in modern vehicles. Topics include identifying hidden frame damage, checking for radiator and condenser leaks, assessing damage to sensors for Advanced Driver-Assistance Systems (ADAS), and criteria for when a bumper can be repaired versus replaced.','Assessing Front-End Collision Damage','CURNT',21,'kb_auto_frontend_damage.html',0,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('POL-10329100','POLICY','CUST-84321','US','2020-06-18','An older, expired version of the homeowners policy for customer CUST-84321. This policy had lower dwelling coverage ($400,000) and did not include the water backup endorsement. It was superseded by POL-10326103.','Homeowners Insurance HO-3 Policy (Expired)','EXPRD',2,'policy_84321_ho3_exp2023.pdf',1,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('ART-10329436','ARTICLE','MOLD-001','CA','2022-09-01','Guidelines for identifying and remediating mold following a water damage event. Differentiates between different types of common household mold. Outlines safety gear (respirators, gloves) and proper containment procedures to prevent spore distribution. Recommends professional remediation for affected areas larger than 10 square feet.','Mold Identification and Remediation Basics','CURNT',8,'kb_mold_remediation.html',0,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('ART-10330945','ARTICLE','AUTO-008','US','2023-11-20','This document outlines the process for recalibrating Advanced Driver-Assistance Systems (ADAS) after a windshield replacement. It specifies which sensors (cameras, LiDAR) are affected and why calibration is critical for features like lane-keeping assist and automatic emergency braking to function correctly.','ADAS Recalibration After Windshield Replacement','CURNT',15,'kb_auto_adas_recalibration.html',0,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('POL-10331073','POLICY','CUST-00871','US','2023-08-15','Personal inland marine policy for high-value items. Covers a collection of antique jewelry valued at $75,000 against all risks, including theft and accidental damage, with no deductible. Requires an updated appraisal every three years.','Personal Articles Floater - Jewelry','ACTIVE',0,'policy_00871_articles.pdf',0,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('POL-11000101','POLICY','CUST-33445','CA','2023-02-20','Renters Insurance HO-4 policy. Personal property coverage is $50,000 with a $500 deductible. Liability coverage is $100,000. Covers named perils such as fire, theft, and water damage from internal sources.','Renters Insurance Policy','ACTIVE',0,'policy_33445_ho4.pdf',0,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('ART-11000102','ARTICLE','FIRE-002','US','2022-05-30','A guide to content cleaning and restoration after a fire. Details techniques for removing soot and smoke odor from various materials including textiles, wood furniture, and electronics. Discusses the use of ozone generators and thermal fogging.','Post-Fire Content Cleaning and Restoration','CURNT',9,'kb_fire_content_cleaning.html',0,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('POL-11000103','POLICY','CUST-67890','US','2021-09-10','Motorcycle insurance policy. Liability limits are 50/100/25. Comprehensive and collision coverage have a combined $1,000 deductible. Includes coverage for custom parts and equipment up to $3,000.','Motorcycle Insurance Policy','ACTIVE',1,'policy_67890_moto.pdf',0,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('ART-11000104','ARTICLE','HAIL-001','US','2023-08-01','Best practices for assessing hail damage to asphalt shingle roofs. Explains how to differentiate between cosmetic and functional damage, identify bruises and fractures, and the importance of checking soft metals like vents and gutters for impact marks.','Assessing Hail Damage on Asphalt Shingles','CURNT',42,'kb_hail_damage_assessment.html',0,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('POL-11000105','POLICY','CUST-19873','US','2020-11-01','Expired auto policy for CUST-19873. This policy had lower liability limits and did not include rental reimbursement. Superseded by POL-10328303.','Comprehensive Auto Policy (Expired)','EXPRD',1,'policy_19873_auto_exp2022.pdf',1,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('ART-11000106','ARTICLE','ELEC-004','US','2022-11-15','This article covers the diagnosis of electrical faults after a power surge event. It describes how to inspect a main breaker panel for tripped breakers, test outlets for power, and identify signs of damage to sensitive electronics and appliances. Emphasizes consulting a licensed electrician for any repairs.','Diagnosing Post-Surge Electrical Issues','CURNT',5,'kb_electrical_surge_diagnosis.html',0,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('POL-11000107','POLICY','CUST-55429','CA','2022-01-05','Previous version of the commercial property policy for CUST-55429. Expired in 2024. Had lower limits for business personal property.','Commercial Property Insurance (Expired)','EXPRD',1,'policy_55429_comm_exp2024.pdf',1,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('ART-11000108','ARTICLE','AUTO-011','US','2024-01-20','Guide to Paintless Dent Repair (PDR) for minor auto body damage, typically caused by hail or door dings. Explains the process, its limitations, and when it is a preferable and cost-effective alternative to traditional bodywork and painting.','Introduction to Paintless Dent Repair (PDR)','CURNT',18,'kb_auto_pdr.html',0,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('POL-11000109','POLICY','CUST-12121','GB','2023-05-01','Home insurance policy for a property in a designated flood plain. Standard policy with a mandatory flood insurance endorsement from a government-backed program. Dwelling coverage is £350,000.','UK Home Insurance with Flood Endorsement','ACTIVE',0,'policy_12121_home_uk.pdf',0,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('ART-11000110','ARTICLE','HVAC-002','US','2021-06-25','Procedure for inspecting an HVAC system for damage after a lightning strike. Checks include the compressor, capacitor, and control board. Advises adjusters to obtain a diagnostic report from a certified HVAC technician to confirm the cause of failure.','Lightning Damage Inspection for HVAC Systems','CURNT',7,'kb_hvac_lightning_damage.html',0,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('POL-11000111','POLICY','CUST-84321','US','2023-06-18','A separate personal umbrella policy providing an additional $1,000,000 in liability coverage over the limits of the primary home (POL-10326103) and auto policies for the insured.','Personal Umbrella Liability Policy','ACTIVE',0,'policy_84321_umbrella.pdf',0,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('ART-11000112','ARTICLE','THEFT-001','US','2023-02-15','This document provides guidance on substantiating claims for stolen personal property. It lists acceptable forms of proof of ownership and value, such as receipts, photographs, appraisals, and credit card statements. Also discusses the role of a police report in the claims process.','Substantiating a Personal Property Theft Claim','CURNT',25,'kb_theft_claim_substantiation.html',0,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('POL-11000113','POLICY','CUST-99988','US','2024-02-01','Condo Unit Owners Insurance (HO-6). Covers personal property ($75,000) and interior finishes/upgrades ($50,000). Also includes liability and loss assessment coverage. The master policy from the condo association covers the building exterior and common areas.','Condo Insurance HO-6 Policy','ACTIVE',0,'policy_99988_ho6.pdf',0,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('ART-11000114','ARTICLE','WIND-001','US','2022-08-20','Technical bulletin on identifying wind vs. water damage on interior ceilings and walls. Indicators of wind damage include uplifted roof decking creating nail pops, while water damage often presents as staining with clear tide lines.','Differentiating Wind vs. Water Damage Internally','CURNT',11,'kb_wind_vs_water_damage.html',0,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('POL-11000115','POLICY','CUST-77766','CA','2022-12-15','Boatowners insurance policy for a 25-foot recreational vessel. Agreed value hull coverage is C$80,000. Includes liability, medical payments, and trailer coverage. Navigational limits apply to the Great Lakes region.','Boatowners Insurance Policy','ACTIVE',1,'policy_77766_boat.pdf',0,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('ART-11000116','ARTICLE','AUTO-015','CA','2023-04-12','Guide to the appraisal clause in Canadian auto policies. Explains the process for resolving disputes over the value of a vehicle or the amount of a loss when the insurer and insured cannot agree. Each party hires an appraiser, and they jointly select an umpire.','Using the Appraisal Clause in Auto Claims (CAN)','CURNT',4,'kb_auto_appraisal_clause_ca.html',0,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('POL-11000117','POLICY','CUST-33445','CA','2021-02-20','Expired renters policy for CUST-33445. Personal property limit was lower at $30,000. Superseded by POL-11000101.','Renters Insurance Policy (Expired)','EXPRD',1,'policy_33445_ho4_exp2023.pdf',1,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('ART-11000118','ARTICLE','SEWER-001','US','2022-10-01','Explanation of coverage for sewer line backups. Clarifies that this is typically excluded from a standard homeowners policy but can be added via a specific endorsement (e.g., water backup and sump pump overflow). Differentiates from flood damage.','Coverage for Sewer Line Backups Explained','CURNT',19,'kb_sewer_backup_coverage.html',0,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('POL-11000119','POLICY','CUST-22334','US','2023-10-20','Classic car insurance policy for a 1968 Ford Mustang. Insured on an agreed value basis of $50,000. Policy has restrictions, including an annual mileage limit of 3,000 miles and required storage in a locked garage.','Classic Car Agreed Value Policy','ACTIVE',0,'policy_22334_classic.pdf',0,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('ART-11000120','ARTICLE','GEN-001','US','2024-02-28','An overview of the principle of indemnity, a core concept in insurance. Explains that the goal of a policy is to restore the insured to the same financial position they were in before the loss occurred, not to provide a financial gain. Discusses Actual Cash Value (ACV) vs. Replacement Cost (RC).','The Principle of Indemnity in Insurance','CURNT',50,'kb_indemnity_principle.html',0,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('POL-11000121','POLICY','CUST-19873','US','2022-11-01','Gap insurance endorsement for the auto policy POL-10328303. Covers the difference between the actual cash value of the vehicle and the amount still owed on the loan or lease in the event of a total loss.','Auto Loan/Lease Gap Insurance','ACTIVE',0,'endorsement_19873_gap.pdf',0,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('ART-11000122','ARTICLE','PLUMB-003','US','2023-09-05','Guide to identifying damage from frozen pipes. Details include signs of pipe bursts, such as visible cracks or slow leaks after thawing, and the types of water damage that typically result. Includes prevention tips for homeowners in cold climates.','Damage Identification from Frozen Pipes','CURNT',6,'kb_frozen_pipe_damage.html',0,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('POL-11000123','POLICY','CUST-88877','GB','2023-07-30','Pet insurance policy for a dog. Covers accidents and illnesses with a £5,000 annual limit. Includes a £100 excess per condition. Excludes pre-existing conditions and routine wellness care.','Pet Accident & Illness Policy','ACTIVE',1,'policy_88877_pet.pdf',0,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('ART-11000124','ARTICLE','ROOF-005','US','2023-01-18','An article discussing matching issues for roof and siding repairs. Explains concepts like reasonable uniformity and line of sight. Outlines state-specific regulations that may require full replacement if a reasonable match cannot be achieved for damaged materials that are no longer available.','Siding and Roof Matching Issues in Claims','CURNT',22,'kb_material_matching.html',0,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('POL-11000125','POLICY','CUST-55429','CA','2024-01-05','Business Interruption endorsement for commercial policy POL-10328505. Provides coverage for lost income and extra expenses incurred if the business must shut down temporarily due to a covered peril like fire.','Business Interruption Endorsement','ACTIVE',0,'endorsement_55429_bi.pdf',0,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('ART-11000126','ARTICLE','AUTO-018','US','2024-03-01','Analysis of catalytic converter theft claims. Details common methods of theft, the types of vehicles most frequently targeted, and replacement costs. Provides loss prevention recommendations for policyholders.','Catalytic Converter Theft Claims','CURNT',14,'kb_catalytic_converter_theft.html',0,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('POL-11000127','POLICY','CUST-99988','US','2024-02-01','Loss Assessment Coverage endorsement for condo policy POL-11000113. Increases coverage to $50,000 for special assessments levied by the condo association for repairs to common areas resulting from a covered loss.','Increased Loss Assessment Coverage','ACTIVE',0,'endorsement_99988_loss_assess.pdf',0,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('ART-11000128','ARTICLE','SUBRO-001','US','2022-04-11','An introduction to the concept of subrogation. Explains the process by which an insurer, after paying a loss, can pursue the at-fault third party to recover the amount paid. Uses an example of a rear-end auto collision.','Introduction to Subrogation','CURNT',30,'kb_subrogation_basics.html',0,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('POL-11000129','POLICY','CUST-11155','US','2023-03-14','General liability policy for a small contracting business. Provides $1,000,000 per occurrence and $2,000,000 aggregate for bodily injury and property damage caused by the business operations. Includes products-completed operations coverage.','Commercial General Liability Policy','ACTIVE',1,'policy_11155_cgl.pdf',0,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('ART-11000130','ARTICLE','STRUCT-002','US','2023-10-02','A guide for adjusters on when to engage a structural engineer. Recommends engineering consultation for claims involving foundation cracks, significant frame damage from vehicle impacts, or concerns about roof truss stability after a fire or major storm.','When to Engage a Structural Engineer','CURNT',10,'kb_structural_engineer_engagement.html',0,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('POL-11000131','POLICY','CUST-67890','US','2023-09-10','Liability-only auto insurance policy for an older, secondary vehicle. Bodily Injury limits are 25/50, and Property Damage is $25k. This policy meets the state minimum financial responsibility requirements.','Basic Liability Auto Policy','ACTIVE',0,'policy_67890_basic_auto.pdf',0,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('ART-11000132','ARTICLE','FIRE-004','US','2023-05-22','Explanation of ALE (Additional Living Expenses) coverage in a homeowners policy. Details what types of expenses are typically covered (e.g., hotel bills, restaurant meals, laundry) when a home is uninhabitable due to a covered loss. Coverage is subject to policy limits.','Understanding Additional Living Expenses (ALE)','CURNT',16,'kb_ale_coverage.html',0,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('POL-11000133','POLICY','CUST-22334','US','2021-03-20','An older, expired classic car policy for CUST-22334. The agreed value was lower at $40,000.','Classic Car Policy (Expired)','EXPRD',0,'policy_22334_classic_exp2023.pdf',1,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('ART-11000134','ARTICLE','AUTO-020','US','2024-04-01','Guideline for determining when a vehicle is a total loss. Discusses the Total Loss Threshold (TLT) which varies by state, and the cost of repair vs. Actual Cash Value (ACV) calculation. Also covers salvage value.','Determining a Vehicle Total Loss','CURNT',28,'kb_auto_total_loss.html',0,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('POL-11000135','POLICY','CUST-84321','US','2023-06-18','Earthquake insurance endorsement for homeowners policy POL-10326103. This optional coverage has a high deductible, typically 15% of the dwelling coverage limit. Covers damage resulting from earth movement.','Earthquake Insurance Endorsement','ACTIVE',0,'endorsement_84321_earthquake.pdf',0,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('ART-11000136','ARTICLE','MOLD-002','US','2023-11-15','An archived article on mold remediation that is now outdated. It has been superseded by ART-10329436, which contains more current industry standards and safety protocols.','Mold Remediation (Archived)','ARCHV',2,'kb_mold_remediation_v1.html',1,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('POL-11000137','POLICY','CUST-77766','CA','2024-05-10','A Personal Watercraft (PWC) policy for two jet skis. Hull coverage is C$15,000 for each unit. Liability is C$500,000. Policy includes coverage for watersports liability.','Personal Watercraft Insurance Policy','ACTIVE',0,'policy_77766_pwc.pdf',0,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('ART-11000138','ARTICLE','GEN-002','US','2024-03-15','A glossary of common insurance terms for adjusters. Defines terms such as peril, hazard, deductible, exclusion, endorsement, and insurable interest. Provides simple examples for each term.','Glossary of Common Insurance Terms','CURNT',45,'kb_insurance_glossary.html',0,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('POL-11000139','POLICY','CUST-11155','US','2023-03-14','Workers Compensation policy for a small contracting business with 5 employees. Provides coverage for medical expenses and lost wages for employees injured on the job, as required by state law.','Workers Compensation Policy','ACTIVE',0,'policy_11155_wc.pdf',0,NULL);
  insert into customer_records_data(id, type, number, country, date, abstract, title, kind, num_claims, filename, withdrawn, abstract_embeddings) values('ART-11000140','ARTICLE','FRAUD-001','US','2024-05-01','An overview of common indicators of potential insurance fraud in property claims. Red flags include claims filed shortly after policy inception, handwritten receipts for expensive items, refusal to allow a scene inspection, and discrepancies in the claimant''s story.','Red Flags for Property Claim Fraud','CURNT',3,'kb_property_fraud_indicators.html',0,NULL);
  ```

When the query has executed successfully, you see a message that says Statement executed successfully.

Task 3. Generate and store text embeddings
------------------------------------------

Now that you have loaded the data, you can generate text embeddings for the abstracts and add them to the `customer_records_data` table.

### Test the query to generate text embeddings

1. In the AlloyDB Studio query editor, click **Clear**.
2. To test the query to generate embeddings using the Vertex AI model, copy and paste the following query, and click **Run**.

  ```
  SELECT embedding('text-embedding-005', 'AlloyDB is a managed, cloud-hosted SQL database service.');
  ```

> **Note:** If you receive a \`Permission denied on the resource\` error, wait a few minutes for the IAM permissions to propagate, and then run the query again.

### Update the table with generated embeddings

1. In the AlloyDB Studio query editor, click **Clear**.
2. To generate embeddings for all abstracts and add them to the `abstract_embeddings` column, run the following query.
  ```
  UPDATE customer_records_data
  SET abstract_embeddings =embedding('text-embedding-005', abstract);
  ```

When the query has executed successfully, you see a message that says Statement executed successfully.

Task 4. Perform vector search in AlloyDB
----------------------------------------

With the text embeddings stored, you can now perform a real-time vector search.

In this task, you run a query to perform a [similarity search](https://cloud.google.com/alloydb/docs/ai/work-with-embeddings#generate-embeddings-similarity-search) to find the most relevant records, even if there isn't an exact match in the text.

1. In the AlloyDB Studio query editor, click **Clear**.
2. To perform a vector search, copy and paste the following query, and click **Run**.
  ```
  SELECT id, title, abstract
  FROM customer_records_data
  ORDER BY abstract_embeddings <=> embedding('text-embedding-005', 'what should I do about water damage in my home?')::vector
  LIMIT 10;
  ```
This query uses the values in the **abstract_embeddings** column to find the top 10 database rows that are the most semantically similar to the search phrase. Notice that the embedding for the search phrase is generated dynamically in the query.

The output resembles the following:

|---|---|---|
| id | title | abstract |
| ART-10326507 | Standard Procedure for Mitigating Water Damage | This article details the standard procedure for mitigating residential water damage... |

Task 5. Optimize vector search with an index
--------------------------------------------

To ensure high-speed queries on large datasets, you can also create a vector index using AlloyDB's implementation of ScaNN (Scalable Nearest Neighbors). In this task, you create an index on customer_records_data, which will automatically be used by the database to accelerate similarity search queries.

1. In the AlloyDB Studio query editor, click Clear.
2. To create the index, copy and paste the following query, and click Run.
  ```
  CREATE INDEX ON customer_records_data
  USING ivfflat (abstract_embeddings vector_l2_ops)
  WITH (lists =100);
  ```

Part 2: The secure API layer
----------------------------


Task 6. Build and deploy the secure API service
-----------------------------------------------

In this task, you deploy the MCP Toolbox for Databases as a secure, private API service. This involves installing the toolbox, creating a custom tools.yaml configuration file, and then building and deploying a new container image to Cloud Run that references the tools.yaml file

### Prepare AlloyDB for the toolbox interaction

1. In Cloud Shell, identify your IP of your Cloud Shell machine by running the following command:
```
ifconfig
```

2. In the eth0 section of the output, copy the inet address value, such as 10.88.0.4.
   The section resembles the following:
   > eth0: flags=4163 ... mtu 1460
   > inet 10.88.0.4  netmask 255.255.0.0  broadcast 10.88.255.255

3. Go back to AlloyDB, and click on the instance named cymbal-instance.
4. In the AlloyDB menu under Primary Cluster, click Connectivity, and then click Edit.
5. Under the Public IP connectivity section, enable the checkbox for Enable public IP.
6. Under Authorized external networks, in the Networks textbox, paste the IP address of your Cloud Shell machine from step 1 as a starting point, but you make some modifications.
   First, replace the last 2 digits with 0.0. Then, add /16 to the end of the address to represent the mask size.
   For example, 10.88.0.4 should be modified to be 10.88.0.0/16.
7. Click Update.

### Secure the foundation (IAM and Secrets)
In this section, you set up the necessary permissions (IAM roles) for a new service account you create for the MCP Toolbox workflow and create a secret to store your AlloyDB database password. This ensures your database password is never exposed and that the service account runs with the minimum required permissions.

1. In Cloud Shell, define variables for the ID and default region of your Google Cloud project:
   ```
   export PROJECT_ID=$(gcloud config get-value project)
   export REGION="REGION"
   ```
2. Create a dedicated service account (the identity for your Cloud Run service) by running the following command:
    ```
    gcloud iam service-accounts create toolbox-identity \
    --display-name="MCP Toolbox Service"
    ```
    Next, you grant some IAM roles to the service account that you just created.

3. To allow the service account to access secrets, run the following command to grant roles/secretmanager.secretAccessor:
    ```
    gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:toolbox-identity@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
    ```
4. To allow the service account to connect to AlloyDB, run the following command to grant roles/alloydb.client:
  ```
  gcloud projects add-iam-policy-binding ${PROJECT_ID} \
      --member="serviceAccount:toolbox-identity@${PROJECT_ID}.iam.gserviceaccount.com" \
      --role="roles/alloydb.client"
  ```
The Service Usage Consumer role is also necessary for any service account that needs to interact with Google Cloud APIs, including the AlloyDB API.

5. To allow the service account to interact with Google Cloud APIs, run the following command to grant roles/serviceusage.serviceUsageConsumer:
    ```
    gcloud projects add-iam-policy-binding ${PROJECT_ID} \
        --member="serviceAccount:toolbox-identity@${PROJECT_ID}.iam.gserviceaccount.com" \
        --role="roles/serviceusage.serviceUsageConsumer"
    ```
6. Last, to grant yourself permission to deploy as the service account, run the following command to grant roles/iam.serviceAccountUser to your student account:
    ```
    gcloud iam service-accounts add-iam-policy-binding \
        toolbox-identity@${PROJECT_ID}.iam.gserviceaccount.com \
        --member="user:$(gcloud config get-value account)" \
        --role="roles/iam.serviceAccountUser"
    ```
    Now, you create the secret and variable to store the AlloyDB password.

7. To store the AlloyDB password securely in Secret Manager, run the following command:
    ```
    echo 'changeme' | gcloud secrets create alloydb-password --data-file=-
    ```
8. To create variable for the secret storing the database password, run the following command:
    ```
    export DB_PASSWORD=$(gcloud secrets versions access latest --secret="alloydb-password")
    ```
    You reference this variable later as {$DB_PASSWORD} when you set up the configuration file for the toolbox.

### Install MCP Toolbox

1. In Cloud Shell, create a new directory named mcp-toolbox and move into that directory:
    ```
    mkdir mcp-toolbox && cd mcp-toolbox
    ```
2. Run the following command to install the binary version of MCP Toolbox for Databases:
    ```
    export VERSION=0.12.0
    curl -O https://storage.googleapis.com/genai-toolbox/v$VERSION/linux/amd64/toolbox
    chmod +x toolbox
    ```

### Deploy MCP Toolbox for Databases
Next, you create the tools.yaml configuration file that defines the database connection and the specific tool the agent can use.

1. In Cloud Shell, create a new file named tools.yaml with the following command:
    ```
    cat << EOF > tools.yaml
    ```
    A new line appears for you to add some code to the file.

2. Copy and paste the following tools.yaml script, and then press Enter:
    ```
    #Define the AlloyDB data source
    sources:
      customer_records_data:
        kind: "alloydb-postgres"
        project: "PROJECT_ID"
        region: "REGION"
        cluster: "cymbal-cluster"
        instance: "cymbal-instance"
        database: "postgres"
        user: "postgres"
        password: "${DB_PASSWORD}"
    
    tools:
      #This tool performs a vector similarity search on customer record abstracts.
      find_similar_customer_records:
        kind: postgres-sql
        source: customer_records_data
        description: "Finds customer records with abstracts similar to a given query text. Use this for semantic search on the contents of policies and articles."
        parameters:
          - name: query_text
            type: string
            description: "The text to search for similar abstracts."
          - name: limit
            type: integer
            description: "The maximum number of similar records to return. Defaults to 5."
        # This statement uses the in-database embedding function and vector search.
        statement: |
          SELECT id, title, abstract, 1 - (embedding('text-embedding-005', $1) <=> abstract_embeddings) as similarity
          FROM customer_records_data
          ORDER BY embedding('text-embedding-005', $1) <=> abstract_embeddings
          LIMIT $2;
    
      #This tool retrieves a specific record by its unique ID.
      get_record_by_id:
        kind: postgres-sql
        source: customer_records_data
        description: "Retrieves the full details of a specific insurance policy or article when the exact record ID is known. Use this for precise lookups."
        parameters:
          - name: record_id
            type: string
            description: "The unique identifier of the policy or article to retrieve (e.g., 'POL-10326103')."
        # This statement performs a direct lookup using the record ID.
        statement: |
          SELECT id, type, number, country, date, title, abstract, kind, num_claims, filename
          FROM customer_records_data
          WHERE id = $1;
    
    toolsets:
      #This groups your tools for easy loading by the agent.
      customer_data_tools:
        - find_similar_customer_records
        - get_record_by_id
    ```

3. On the new line, type EOF and press Enter again to exit the editor, and return to the command line.
4. Within the current working directory named mcp-toolbox, run the following command to start the server for MCP Toolbox for Databases:
    ```
    ./toolbox --tools-file "tools.yaml" --port=8080
    ```
    Leave this terminal window open, so that the service keeps running.

5. [Optional] You can test that the service is running by clicking Web Preview (along the menu bar for Cloud Shell next to Open Editor), and selecting Preview on port 8080.

This action opens the service in a new browser tab with the message Hello World!. After confirming that you see the message Hello World!, close the browser tab. 

### Deploy MCP Toolbox for Database to Cloud Run
1. Open a new Cloud Shell terminal window by clicking + for Open a new tab (top of the terminal window next to the project ID ).

2. In the new Cloud Shell terminal window, navigate to the directory named mcp-toolbox:
    ```
    cd mcp-toolbox
    ```

3. Recreate the variables for REGION and PROJECT_ID in this new terminal window:
    ```
    export PROJECT_ID=$(gcloud config get-value project)
    export REGION="REGION"
    ```
4. Create a new secret based on your tools.yaml file:
    ```
    gcloud secrets create tools --data-file=tools.yaml
    ```
  If you get an error that the secret already exists, you can update it with the following command, but otherwise, you can proceed to step 5:
    ```
    gcloud secrets versions add tools --data-file=tools.yaml
    ```

5. Next, run the following command to ensure that the AlloyDB instance and Cloud Run are on the same VPC:
    ```
    gcloud compute networks vpc-access connectors create alloydb-connector \
        --region=${REGION} \
        --network=peering-network \
        --range=10.8.0.0/28
    ```
6. Create a variable for the container image to be used for Cloud Run:
    ```
    export TARGET_IMAGE=us-central1-docker.pkg.dev/database-toolbox/toolbox/toolbox:latest
    ```
  Next, you deploy the MCP Toolbox for Database to Cloud Run. The code below deploys the pre-built toolbox container, injects your tools.yaml as a secret, and is configured to run securely.

7. Run the following to deploy MCP Toolbox for Database to Cloud Run:
    ```
    gcloud run deploy toolbox \
        --image=${TARGET_IMAGE} \
        --vpc-connector=alloydb-connector \
        --region=${REGION} \
        --service-account=toolbox-identity \
        --no-allow-unauthenticated \
        --set-secrets="/app/tools.yaml=tools:latest" \
        --args="--tools-file=/app/tools.yaml","--address=0.0.0.0","--port=8080" \
        --ingress=internal \
        --min-instances=1
    ```
  The deployment process will take several minutes. Once complete, your secure API service will be running.

8. After deployment finishes, copy the Service URL (such as https://toolbox-1039773418264.us-east4.run.app), and run the following command to store it in an variable by replacing PASTE_SERVICE_URL_HERE with the URL you just copied:
  ```
  export SECURE_API_URL="PASTE_SERVICE_URL_HERE"
  ```

Part 3: The AI Agent Layer
--------------------------


Task 7. Set up the ADK agent environment
----------------------------------------

Now, you can set up a proper development environment for the agent that securely consumes your new API.

1. In same Cloud Shell session from the previous section (in which you deployed Cloud Run), create a new directory for your agent:
    ```
    mkdir -p my-adk-agent/multi-tool-agent && cd my-adk-agent
    ```
2. Next, create and activate a Python virtual environment to isolate your agent's dependencies:
    ```
    python3 -m venv .venv
    source .venv/bin/activate
    ```
    Your command prompt should now be prefixed with (.venv).

3. Last, install the ADK library:
    ```
    pip install google-adk toolbox-core
    ```

Task 8. Build the ADK agent
---------------------------
In this task, you use adk to create the initial structure and files for your agent using adk create and name your agent app multi_tool_agent.

1. In the same Cloud Shell terminal from the previous task, run the following to create the agent app named multi_tool_agent:
    ```
    adk create multi_tool_agent
    ```
2. When prompted to choose a model for the root agent, enter 1 for gemini-2.5-flash.

3. When prompted to choose a backend, enter 2 for Vertex AI.

4. When prompted to enter a Google Cloud project ID, enter:

6. When prompted to enter a Google Cloud region, enter:

The output resembles the following:
>Agent created /home/student_04_abe4bdb38f4f/mcp-toolbox/my-adk-agent/multi_tool_agent:
>  - .env
>  - __init__.py
>  - agent.py

### Test the simple agent
    ```
    adk web
    
    # select multi_tool_agent
    
    # Type Hello
    
    # Close the session
    ```
### Update the agent definition
Now you can create the script that defines your agent's properties and, crucially, links to the customer_data_tools toolset you defined in tools.yaml.

1. Within the current working directory named my-adk-agent, copy the tool configuration file named tools.yaml that you previously created for the API layer into your agent's directory, so the ADK can discover it:
    ```
    cp ../tools.yaml .
    ```
2. Update the file named agent.py:
    ```
    cat << EOF > agent.py
    ```
    A new line appears for you to add code to the file.

3. Copy and paste the following agent.py script, and then press Enter:
    ```
    from google.adk.agents import Agent
    from toolbox_core import ToolboxSyncClient
    
    #Load all the tools
    toolbox = ToolboxSyncClient(${SECURE_API_URL})
    tools = toolbox.load_toolset('customer_data_tools')
    
    #Define the agent at the module level and assign it to root_agent
    root_agent = Agent(
        name='claims_assistant',
        model='gemini-2.5-flash',
        description= 'The Cymbal Claims Assistant is designed to help insurance adjusters at Cymbal Insurance find relevant articles or policies and find a specific policy or article by providing its unique ID.',
        instruction= 'You are an insurance claims assistant specifically helping insurance adjusters at Cymbal Insurance. Your primary function is to quickly and accurately retrieve information from a database of insurance policies and related knowledge base articles. You streamline the claims process by allowing an adjuster to 1) perform semantic searches using natural language to find relevant articles or policies (e.g., "find procedures for mitigating water damage"); and 2) retrieve the exact details of a specific policy or article by providing its unique ID.',
        tools=tools,
    )
    
    #client_headers={"Authorization": f"Bearer {auth_token}"}
    ```
4. On the new line, type EOF and press Enter again to exit the editor, and return to the command line.

Task 9. Run and test your agent
-------------------------------
You are now ready to run the agent and interact with your secure, three-tier architecture using the ADK's local web UI. Your current working directory named my-adk-agent directory now contains the tools.yaml file and the multi-tool-agent subdirectory with the updated agent.py.

>my-adk-agent/
> - ├── multi_tool_agent/
> - │   ├── __init__.py
> - │   ├── agent.py
> - │   └── .env
> - └── tools.yaml

1. Within the current working directory named my-adk-agent, run the following command to launch the ADK Web UI:
    ```
    adk web
    
    # select multi_tool_agent
    
    # ask the agent questions like, "Find articles about roof damage from storms."
    
    ```



[1](https://www.skills.google/paths/3273/course_templates/1436/labs/586353)
