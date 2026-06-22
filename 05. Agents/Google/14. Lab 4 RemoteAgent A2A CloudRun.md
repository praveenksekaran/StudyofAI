# Deploy to Cloud Run

# Overview

The [Agent2Agent (A2A) protocol](https://a2a-protocol.org/) addresses a critical challenge in the AI landscape: enabling Gen AI agents, built on diverse frameworks by different companies running on separate servers, to communicate and collaborate effectively - as agents, not just as tools. A2A aims to provide a common language for agents, fostering a more interconnected, powerful, and innovative AI ecosystem.

A2A is built around a few core concepts that make it powerful and flexible:

Standardized Communication: JSON-RPC 2.0 over HTTP(S).
Agent Discovery: Agent Cards detail an agent's capabilities and connection info, so agents can discover each other and learn about each other's capabilities
Rich Data Exchange: Handles text, files, and structured JSON data.
Flexible Interaction: Supports synchronous request/response, streaming (SSE), and asynchronous push notifications.
Enterprise-Ready: Designed with security, authentication, and observability in mind.

# Objectives
In this lab, you will:

Deploy an ADK agent as an A2A Server.
Prepare a JSON Agent Card to describe an A2A agent's capabilities.
Enable another ADK agent to read the Agent Card of your deployed A2A agent and use it as a sub-agent.

# Task 1. Install ADK and set up your environment

In this lab environment, the **Vertex AI API and Cloud Run API** have been enabled for you. If you were to follow these steps in your own project, you would enable them by navigating to Vertex AI and following the prompt to enable it.

#### Prepare a Cloud Shell Editor tab

Open Cloud shell and execute cloudshell workspace ~

#### Download and install the ADK and code samples for this lab

1. Install ADK by running the following command in the Cloud Shell Terminal. Note: You will specify the version to ensure that the version of ADK that you install corresponds to the version used in this lab:
  ```
  # Install ADK and the A2A Python SDK
  cd ~
  export PATH=$PATH:"/home/${USER}/.local/bin"
  python3 -m pip install google-adk==1.8.0 a2a-sdk==0.2.16
  pip install --upgrade google-genai
  # Correcting a typo in this version
  sed -i 's/{a2a_option}"/{a2a_option} "/' ~/.local/lib/python3.12/site-packages/google/adk/cli/cli_deploy.py
  ```

2. Paste the following commands into the Cloud Shell Terminal to copy lab code from a Cloud Storage bucket and unzip it:
  ```
  gcloud storage cp gs://YOUR_GCP_PROJECT_ID-bucket/adk_and_a2a.zip ./adk_and_a2a.zip
  unzip adk_and_a2a.zip
  ```

# Task 2. Explore the ADK agent you will make available remotely

For the purposes of this lab, imagine you work for a stadium maintenance company: Cymbal Stadiums. As part of a recent project, you developed an image generation-agent that can create illustrations according to your brand guidelines. Now, several different teams in your organization want to use it too.

If you were to copy the code for use as a sub-agent by many agents, it would be very difficult to maintain and improve all of these copies.

Instead, you can deploy the agent once as an agent wrapped with an A2A server, and the other teams' agents can incorporate it by querying it remotely.

1. In the Cloud Shell Editor's file explorer pane, navigate to the adk_and_a2a/illustration_agent directory. This directory contains the ADK agent you will make available remotely. Click the directory to toggle it open.

2. Open the agent.py file on this directory and scroll to the section labeled # Tools.

3. Notice the generate_image() function, which will be used as a tool by this agent. It receives a prompt and performs a two-step process. First, it uses the Google Gen AI SDK to call generate_content(), which returns the raw image data directly in the response. Second, the function uses the Cloud Storage library to upload these image bytes to a GCS bucket. Finally, the tool returns the public URL of the newly created image file.

4. Notice that the instruction provided to the root_agent provides specific instructions to the agent to use image-generation prompts that respect the company's brand guidelines. For example, it specifies:
  - a specific illustration style: (Corporate Memphis)
  - a color palette (purples and greens on sunset gradients)
  - examples of stadium/sports and maintenance imagery because it is a stadium maintenance company

5. To see it in action, you'll first need to write a .env file to set environment variables needed by ADK agents. Run the following in the Cloud Shell Terminal to write this file in this directory.
```
cd ~/adk_and_a2a
cat << EOF > illustration_agent/.env
GOOGLE_GENAI_USE_VERTEXAI=TRUE
GOOGLE_CLOUD_PROJECT=YOUR_GCP_PROJECT_ID
GOOGLE_CLOUD_LOCATION=global
MODEL=gemini_flash_model_id
IMAGE_MODEL=gemini_flash_image_model_id
EOF

# Copy .eve to other folders
cp illustration_agent/.env slide_content_agent/.env

# Run the agent
adk web

# Select  illustration_agent and type: By supporting each other, we get big things done!

# Image URL will be provided after 10 seconds
```

6. Notice that the prompt you provided to the agent didn't mention sports, stadiums, or maintenance work, but the agent took your text and the brand guidelines and combined them into a single prompt for the image generation model.

7. Click on the Cloud Shell Terminal pane and press CTRL + C to stop the server

# Task 3. Deploy the agent as an A2A Server
You'll now take the steps to deploy this agent as a remote A2A agent.

1. An A2A Agent identifies itself and its capabilities by serving an [Agent Card](https://a2a-protocol.org/latest/topics/agent-discovery/#the-role-of-the-agent-card). Run the following to create an agent.json file.
  ```
  touch illustration_agent/agent.json
  ```

2. Open the agent.json file within the adk_and_a2a/illustration_agent directory and paste in the following contents:
  ```
  {
      "name": "illustration_agent",
      "description": "An agent designed to generate branded illustrations for Cymbal Stadiums.",
      "defaultInputModes": ["text/plain"],
      "defaultOutputModes": ["application/json"],
      "skills": [
      {
          "id": "illustrate_text",
          "name": "Illustrate Text",
          "description": "Generate an illustration to illustrate the meaning of provided text.",
          "tags": ["illustration", "image generation"]
      }
      ],
      "url": "https://illustration-agent-Project Number.GCP_LOCATION.run.app/a2a/illustration_agent",
      "capabilities": {},
      "version": "1.0.0"
  }
  ```
3. Save the file.

4. Review the JSON in the agent.json file. Notice that it gives the agent a name and description and identifies some skills . It also indicates a url where the agent itself can be called.

The agent's url is constructed to be its Cloud Run service URL once you have deployed it following the instructions in this lab.

While similar in name to skills, the parameter capabilities here is reserved to indicate abilities like streaming.

5. Run the following to create a requirements.txt file in the illustration_agent directory.
  ```
  touch illustration_agent/requirements.txt
  ```
6. Select the file, and paste the following into the file.
  ```
  google-adk==1.8.0
  a2a-sdk==0.2.16
  ```
7. Save the file.
8. In the following command, you will use adk deploy cloud_run with the --a2a flag to deploy your agent to Cloud Run as an A2A server. You can learn more about deploying agents to Cloud Run by searching for the lab "Deploy ADK agents to Cloud Run". In this command:

- the --project and --region define the project and region in which your Cloud Run service will be deployed
- the --service_name defines the name for the Cloud Run service
- the --a2a flag indicates it should be hosted as an A2A agent. This means two things:
  - your agent will be wrapped by a class which bridges ADK and A2A agents: the [A2aAgentExecutor](https://github.com/google/adk-python/blob/main/src/google/adk/a2a/executor/a2a_agent_executor.py). This class translates A2A Protocol's language of [tasks and messages](https://a2a-protocol.org/latest/topics/key-concepts/#fundamental-communication-elements) to an ADK Runner in its language of events.
  - the Agent Card will be hosted as well at **CLOUD_RUN_URL/a2a/AGENT_NAME/.well-known/agent.json**. Note: While this version of the card will be usable soon, the dynamic rewriting of the agent's url currently does not work with Cloud Run, so we won't use it in this version of this lab.

Deploy the agent to Cloud Run as an A2A server with the following command:
  ```
  adk deploy cloud_run \
    --project YOUR_GCP_PROJECT_ID \
    --region GCP_LOCATION \
    --service_name illustration-agent \
    --a2a \
    illustration_agent
  ```
9. You will be prompted to allow unauthenticated responses for this container. For the sake of lab testing, enter Y into the Cloud Shell Terminal (for "yes") and press return.
> Note: Deployment should take about 5-10 minutes. If you encounter a PERMISSION_DENIED error, try running the above command again.

# Task 4. Enable another ADK agent to call this agent remotely
In this task, you will provide a second ADK agent the ability to identify your illustration agent's capabilities and call it remotely. This second agent will be an agent tasked with creating contents for slides. It will write a headline and a couple of sentences of body text, then transfer to the illustration agent to generate an image to illustrate that text.

1. In the Cloud Shell Terminal, run the following command to copy the Agent Card JSON file to your adk_and_a2a directory and change its name to indicate that it represents the illustration_agent.
  ```
  cp illustration_agent/agent.json illustration-agent-card.json
  ```
2. In the Cloud Shell Editor's file explorer pane, navigate to the adk_and_a2a/slide_content_agent and open the agent.py file.

Review this agent's instruction to see it will take a user's suggestion for a slide and write a headline & body text, then transfer to your A2A agent to illustrate the slide.

3. Paste the following code under the # Agents header to add the remote agent using the [RemoteA2aAgent](https://github.com/google/adk-python/blob/main/src/google/adk/agents/remote_a2a_agent.py) class from ADK:
  ```
  illustration_agent = RemoteA2aAgent(
      name="illustration_agent",
      description="Agent that generates illustrations.",
      agent_card=(
          "illustration-agent-card.json"
      ),
  )
  ```

4. Add the illustration_agent as a sub-agent of the root_agent by adding the following parameter to the root_agent:
  ```
  sub_agents=[illustration_agent]
  ```

5. Save the file.
6. Launch the UI from the Cloud Shell Terminal with:
```
cd ~/adk_and_a2a
adk web

# A new browser tab will open with the ADK Dev UI. From the Select an agent dropdown on the left, select the slide_content_agent from the dropdown.

# Query the agent with an idea for a slide:
Create content for a slide about our excellent on-the-job training.
```

