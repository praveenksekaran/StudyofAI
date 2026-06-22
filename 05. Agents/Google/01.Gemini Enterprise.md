Gemini Enterprise is an **AI-powered hub** that orchestrates Gemini models, agents, and your enterprise data to give grounded answers and execute 
workflows from a single prompt.[1]

# Introduction 

## Core idea

- Central command center that connects Google-quality search, Gemini reasoning, and enterprise data (wherever hosted) to surface the right information at the
- right time.[1]
- Helps employees do planning, research, content generation, and action execution through conversational prompts, significantly boosting productivity.[1]

## What AI agents do

- **Definition:** Smart software assistants that plan, reason, and act to help with information retrieval, task automation, data analysis, and communication.[1]
- Key capabilities:[1]
  - Information retrieval and summarization: Find specific info in large datasets and generate concise summaries.  
  - Task automation: Execute actions, trigger workflows, send notifications, update records.  
  - Data analysis and reporting: Analyze data, find patterns, create reports/visuals.  
  - Communication and interaction: Natural language conversations, FAQs, customer support.  

## Types of agents in Gemini Enterprise

- **Google-built agents** (ready-made):[1]
  - The Assistant (Gemini LLM): broad requests, text generation, everyday tasks.  
  - Search engine: retrieves information from vast data sources.  
  - NotebookLM: conducts deep research over user-provided sources.  

- **No-code agents** (for business users):[1]
  - Visual builder to design and launch custom AI helpers without coding.  
  - Used to quickly create automations tailored to specific business problems.  

- **High-code agents** (for developers):[1]
  - Full coding capabilities to implement complex logic and deep integrations.  
  - Can use enterprise and external data, advanced models (e.g., Gemini) for highly customized intelligence.  

- **Third-party agents**:[1]
  - Integrate with systems like Salesforce, Jira, SharePoint, Microsoft Copilot.  
  - Use prebuilt connectors that respect original ACLs, unifying data while preserving access controls.  

## Gemini Enterprise workflow

- Acts as a hub that:[1]
  - Takes a user prompt and orchestrates internal AI capabilities plus custom agents.  
  - Connects to internal and external data sources and fulfillment systems (email, ticketing, databases).  
  - Returns grounded, relevant, and actionable responses, including executing workflows.  

## Putting it together / exam focus

- Primary purpose: **Empower employees to find information, generate personalized answers, and perform tasks by integrating with workflow actions
  and enterprise data.**[1]
- Remember contrasts:[1]
  - It augments, not replaces, existing data stores.  
  - It uses both Google-built and custom/third-party agents.  
  - It serves organization-wide use cases, not just personal schedule management.

[1](https://www.skills.google/paths/3273/course_templates/1401/documents/606703)

# NotebookLLM
NotebookLM acts as a virtual research assistant, helping you synthesize information across multiple sources into organized insights. 
You can upload documents like PDFs, Google Docs, Google Slides, text files, images, and audio files, along with website URLs and YouTube videos, 
to build a notebook with many sources.

In essence, NotebookLM is your personal or team-specific expert on a defined corpus of information, helping you deeply understand and synthesize it. 
Gemini Enterprise is your enterprise-wide AI brain and automation hub, connecting all your data and applications to empower intelligent search and complex, 
action-oriented workflows through specialized agents.

## What's the difference between NotebookLM and Gemini Enterprise?

While both Gemini Enterprise and NotebookLM utilize RAG architecture to connect with data, their scope and functionalities diverge significantly. 
Gemini Enterprise is designed for a broad, enterprise-wide reach, connecting to a vast array of enterprise data and applications, offering extensive support 
for various data types. In contrast, NotebookLM focuses on user- or team-specific data, with a more limited range of supported data types

# Gemini Enterprise Homepage 
Gemini Enterprise provides a **customizable homepage** that unifies enterprise search, generative AI, and workflow actions so employees can quickly find information and get work done.[1]

## Why the homepage matters

- Enterprises lose time because information is scattered across docs, email, chat, tickets, and databases; much of what employees retrieve is irrelevant.[1]
- Most employees search across many sources, struggle to find what they need, and prefer a Google‑like search experience.[1]

## Homepage layout: key sections

- **Search bar:** Central entry point for searching, asking questions, and triggering actions.[1]
- **Agents:** Purpose-built helpers for specific tasks; users can use provided agents or custom ones.[1]
- **Prompts:** Example queries you can click to quickly try common use cases.[1]

- **For you widgets:**[1]
  - Calendar widget: upcoming meetings.  
  - Drive widget: quick access to recent documents.  

- **Announcements:** Organization-wide updates surfaced directly on the homepage.[1]
- **Additional widgets:** Extra panels depending on which services are connected to the app.[1]

## What the search bar can do

- Search across content sources (Drive, Gmail, other systems) for documents and emails.[1]
- Search within documents for specific, detailed information (for example, Q4 sales by city).[1]
- Summarize documents with generative AI and provide links to the source files.[1]
- Answer specific questions using combined data you already have access to, with suggested follow‑up questions.[1]
- Provide guidance and brainstorming help for tasks like preparing presentations.[1]
- Take actions in connected apps (Gmail, Calendar, others), such as drafting emails or creating events for your review.[1]
- Connect users into custom agent workflows defined by your team.[1]

## Access, deployment, and URL

- Gemini Enterprise is configured and deployed as an app within AI Applications.[1]
- After deployment, you get a public URL to share with users or map to a company subdomain such as home.mycompany.com using DNS.[1]

## Data privacy and security

- Gemini Enterprise does not grant new permissions; it only surfaces content you already have access to.[1]
- Other users only see information in documents that are shared with them.[1]
- Your information is not used to train models for Google or other Google customers.[1]

## Module takeaway (exam focus)

- Core value: **Connect content across the organization, generate grounded personalized answers, and perform tasks via integrated workflows**, all from a single homepage.[1]
- It solves knowledge discovery problems by combining Google search expertise, generative AI, and cloud infrastructure into a customizable, expandable enterprise hub.[1]

[1](https://www.skills.google/paths/3273/course_templates/1191/documents/594554)

# Prepare The Env with APIs, IAM and Auth settings 

Configuring Gemini Enterprise focuses on **APIs, IAM roles, and authentication** so the app can enforce existing access controls securely.[1]

## Required API and admin roles

- Enable the **Discovery Engine API** on the Google Cloud project that will host the AI Applications app.[1]
- The configurator needs these roles:[1]
  - **Discovery Engine Admin** (manage app, data stores, actions)  
  - **OAuth Config Editor** (set up OAuth consent and client for actions)  
  - **Service Usage Admin** (enable required APIs)  

## End-user access role

- End users of the Gemini Enterprise homepage must have the **Discovery Engine User** IAM role to perform search and assistant actions.[1]
- This role grants only the permissions needed to use the app, not to manage configuration.[1]

## Access control and identity provider

- Gemini Enterprise enforces access using your existing identity provider and its **ACLs**, so users see only what they already have access to in docs, calendars, and other data.[1]
- You can use **Google Identity** or external IdPs that support **OpenID Connect (OIDC)** or **SAML 2.0** (for example, Azure AD, Okta) via **Workforce Identity Federation (WIF)**.[1]

## Authentication configuration

- Connect and configure your identity provider within AI Applications so the Gemini Enterprise app can authenticate users consistently.[1]
- Once configured, authentication and authorization mirror current enterprise patterns, avoiding separate permission models just for Gemini Enterprise.[1]

[1](https://www.skills.google/paths/3273/course_templates/1191/documents/594555)

# Connecting App to Data store 

This chapter explains how to use a **Google Drive data store** in Gemini Enterprise so users can search and query their existing Drive content securely.[1]

## Data stores and RAG

- AI Applications data stores connect Gemini Enterprise to enterprise data and automatically handle chunking, embeddings, and indexing.[1]
- This underpins **Retrieval-Augmented Generation (RAG)**, where relevant chunks are retrieved at query time and used to ground generative answers.[1]

## Purpose of a Google Drive data store

- Lets users search only documents they already have access to in Google Drive, respecting existing permissions.[1]
- Enables the assistant to summarize documents and answer questions directly from their Drive content.[1]

## High-level deployment steps

- Create a Gemini Enterprise app in AI Applications.[1]
- Select the appropriate service tier for the app.[1]
- Add a Google Drive data store as one of the app’s data sources.[1]
- Specify the Drive source for the data store (for example, particular Drives or scopes).[1]

[1](https://www.skills.google/paths/3273/course_templates/1191/documents/594556)

