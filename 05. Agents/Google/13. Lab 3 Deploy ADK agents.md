# Deploy agents to Agent Engine

Vertex AI Agent Engine (formerly known as LangChain on Vertex AI or Vertex AI Reasoning Engine) is a fully managed Google Cloud service enabling developers to deploy, manage, and scale AI agents in production.

You can learn more about its benefits in the [Vertex AI Agent Engine documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/overview).

# Task 1. Install ADK and set up your environment

#### Enable Vertex AI recommended APIs

> In this lab environment, the Vertex AI API has been enabled for you. If you were to follow these steps in your own project, you could enable it by navigating to Vertex AI and following the prompt to enable it

#### Prepare a Cloud Shell Editor tab

1. With your Google Cloud console window selected, open Cloud Shell by pressing the G key and then the S key on your keyboard. Alternatively, you can click the Activate Cloud Shell button (Activate Cloud Shell) in the upper right of the Cloud console.
2. In the Cloud Shell Terminal, enter the following to open the Cloud Shell Editor to your home directory:
  ```
  cloudshell workspace ~
  ```

#### Download and install ADK and code samples for this lab

  ```
  #1. Update your PATH environment variable and install ADK by running the following commands in the Cloud Shell Terminal.
  export PATH=$PATH:"/home/${USER}/.local/bin"
  python3 -m pip install google-adk
  
  # 2. Paste the following commands into the Cloud Shell Terminal to copy a file from a Cloud Storage bucket, and unzip it, creating a project directory with code for this lab:
  gcloud storage cp -r gs://YOUR_GCP_PROJECT_ID-bucket/adk_to_agent_engine .
  
  # 3. Install additional lab requirements with:
  python3 -m pip install -r adk_to_agent_engine/requirements.txt
  
  #4. Run the following commands to create a .env file in the adk_to_agent_engine directory. (Note: To view a hidden file beginning with a period, you can use the Cloud Shell Editor menus to enable View > Toggle Hidden Files):
  cd ~/adk_to_agent_engine
  cat << EOF > .env
  GOOGLE_GENAI_USE_VERTEXAI=TRUE
  GOOGLE_CLOUD_PROJECT=YOUR_GCP_PROJECT_ID
  GOOGLE_CLOUD_LOCATION=GCP_LOCATION
  MODEL=gemini-2.5-flash
  EOF
  
  #5. Copy the .env file to the agent directory to provide your agent necessary authentication configurations once it is deployed:
  cp .env transcript_summarization_agent/.env
  
  ```

# Task 2. Deploy to Agent Engine using the command line deploy method

ADK's command line interface provides shortcuts to deploy agents to Agent Engine, Cloud Run, and Google Kubernetes Engine (GKE). You can use the following base commands to deploy to each of these services:

- [adk deploy agent_engine](https://github.com/google/adk-python/blob/c52f9564330f0c00d82338cc58df28cb22400b6f/src/google/adk/cli/cli_tools_click.py#L1037) (with its command line args described under the **@deploy.command("agent_engine")** decorator)
- [adk deploy cloud_run](https://github.com/google/adk-python/blob/c52f9564330f0c00d82338cc58df28cb22400b6f/src/google/adk/cli/cli_tools_click.py#L861) (with its command line args described under the **@deploy.command("cloud_run")** decorator)
- [adk deploy gke](https://github.com/google/adk-python/blob/c52f9564330f0c00d82338cc58df28cb22400b6f/src/google/adk/cli/cli_tools_click.py#L1192) (with its command line args described under the **@deploy.command("gke")** decorator)

The **adk deploy agent_engine** command wraps your agent in a [reasoning_engines.AdkApp](https://cloud.google.com/python/docs/reference/vertexai/latest/vertexai.preview.reasoning_engines.AdkApp#vertexai_preview_reasoning_engines_AdkApp) class and deploys this app to Agent Engine's managed runtime, ready to receive agentic queries.

When an AdkApp is deployed to Agent Engine, it automatically uses a [VertexAiSessionService](https://google.github.io/adk-docs/sessions/session/#sessionservice-implementations) for persistent, managed session state. This provides multi-turn conversational memory without any additional configuration. For local testing, the application defaults to a temporary, InMemorySessionService.

To deploy an Agent Engine app using adk deploy agent_engine, complete the following steps:

1. In the adk_to_agent_engine/transcript_summarization_agent directory, click on the agent.py file to review the instructions of this simple summarization agent.

2. To deploy an agent, you must provide its requirements. In Cloud Shell Editor, right-click on the transcript_summarization_agent directory. (You may need to click Allow to enable the right-click menu.)

3. Select New File...

4. Name the file like a standard Python requirements file: requirements.txt

5. Paste the following into the file:
  ```
  google-cloud-aiplatform[adk,agent_engines]==1.110.0
  ```

6. Save the file.
7. In the Cloud Shell Terminal, run the deploy command:
  ```
  adk deploy agent_engine transcript_summarization_agent \
  --display_name "Transcript Summarizer" \
  --staging_bucket gs://YOUR_GCP_PROJECT_ID-bucket
  ```
  
  You can follow the status from the log file that will be linked from the command's output. During deployment, the following steps are occurring:
  1. A bundle of artifacts is generated locally, comprising:
     -  *.pkl: a pickle file corresponding to local_agent.
     -  [requirements.txt](https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/deploy#package-requirements): this file from the agent folder defining package requirements.
     -  [dependencies.tar.gz](https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/deploy#extra-packages): a tar file containing any extra packages.
   2. The bundle is uploaded to Cloud Storage (using a defined [directory](https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/deploy#gcs-directory) if specified) for staging the artifacts.
   3. The Cloud Storage URIs for the respective artifacts are specified in the [PackageSpec](https://cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1/projects.locations.reasoningEngines#PackageSpec).
    4. The Vertex AI Agent Engine service receives the request and builds containers and spins up HTTP servers on the backend.

Note: Deployment should take about 10 minutes, but you can continue with this lab while it deploys.

#### Quiz While Your Agent is Deploying

Each of the adk deploy ... commands requires certain arguments to be set. For the most up-to-date arguments, click the linked commands in the list at the top of this task and look for the arguments marked as "Required".

Some required arguments, like [--project](https://github.com/google/adk-python/blob/c52f9564330f0c00d82338cc58df28cb22400b6f/src/google/adk/cli/cli_tools_click.py#L1042) and [--region](https://github.com/google/adk-python/blob/c52f9564330f0c00d82338cc58df28cb22400b6f/src/google/adk/cli/cli_tools_click.py#L1050) from the adk deploy agent_engine deployment can load their values from the agent's .env file if present.

Answer the following questions based on the arguments for [adk deploy agent_engine](https://github.com/google/adk-python/blob/c52f9564330f0c00d82338cc58df28cb22400b6f/src/google/adk/cli/cli_tools_click.py#L1037)

> The `--agent_engine_id` argument allows you to update an existing Agent Engine instance.
> True

> The `--trace_to_cloud` argument has a default value of True.
> False

> Which of the following is true about the `--adk_app` argument?
> Accepts a filename to define an ADK app
 

# Task 3. Get and query an agent deployed to Agent Engine
To query the agent, you must first grant it the authorization to call models via Vertex AI.

1. To see the service agent and its assigned role, navigate to **IAM** in the console.

2. Click the checkbox to **Include Google-provided role grants**.

3. Find the **AI Platform Reasoning Engine Service Agent** (service-PROJECT_NUMBER@gcp-sa-aiplatform-re.iam.gserviceaccount.com), and click the edit pencil icon in this service agent's row.

4. Click **+ Add another role**.

5. In the **Select a role field**, enter **Vertex AI User**. If you deploy an agent that uses tools to access other data, you would grant access to those systems to this service agent as well.

6. Save your changes.

7. Back in the Cloud Shell Editor, within the **adk_to_agent_engine** directory, open the file **query_agent_engine.py**.

8. Review the code and comments to notice what it is doing.

> What does the code in this file do? Select all that apply.
> Establish a logging client
> Initialize Vertex AI
> Load Agent Engine apps, filtering by display name
> Create a session
> Query the summarizer agent

9. Review the transcript passed to the agent, so that you can evaluate if it's generating an adequate summary.
  
10. In the Cloud Shell Terminal, run the file from the **adk_to_agent_engine** directory with:
  ```
  cd ~/adk_to_agent_engine/transcript_summarization_agent
  python3 query_agent_engine.py
  ```

# Task 4. View and delete agents deployed to Agent Engine

1. When your agent has completed its deployment, return to a browser tab showing the Cloud Console and **navigate to Agent Engine** by searching for it and selecting it at the top of the Console.

2. In the **Region** dropdown, make sure your location for this lab **(GCP_LOCATION)** is selected.

3. You will see your deployed agent's display name. Click on it to enter its monitoring dashboard.

4. Notice both the **Metrics and Session** tabs that will each give you insights into how your agent is being used.

5. When you are ready to delete your agent, select **Deployment details** from the top of its monitoring dashboard.

6. Back in your browser tab running the Cloud Shell Terminal, **paste the following command, but don't run it yet:**
   ```
   cd ~/adk_to_agent_engine
    python3 agent_engine_utils.py delete REPLACE_WITH_AE_ID
    ```
7. From the Agent Engine **Deployment info panel**, copy the Name field, which will have a format like: *projects/qwiklabs-gcp-02-76ce2eed15a5/locations/us-central1/reasoningEngines/1467742469964693504*

8. Return to the Cloud Shell Terminal and replace the end of the command REPLACE_WITH_AE_ID with the resource name you've copied
9. Press Return to run the deletion command.
10. In the Cloud Console, return to the Agent Engine dashboard to see that the agent has been deleted.
11. To view the simple Python SDK code to list and delete agents, view the contents of the file **adk_to_agent_engine/agent_engine_utils.py**
