# Use Model Context Protocol (MCP) Tools with ADK Agents[1]

## Overview[1]

In this lab, you will explore [Model Context Protocol](https://modelcontextprotocol.io/) (MCP), an open standard that enables seamless integration between external services, data sources, tools, and applications. You will learn how to integrate MCP into your ADK agents, using tools provided by existing MCP servers to enhance your ADK workflows. Additionally, you will see how to expose ADK tools like `load_web_page` through a custom-built MCP server, enabling broader integration with MCP clients.[1]

**What is Model Context Protocol (MCP)?**[1]

Model Context Protocol (MCP) is an open standard designed to standardize how Large Language Models (LLMs) like Gemini and Claude communicate with external applications, data sources, and tools. Think of it as a universal connection mechanism that simplifies how LLMs obtain context, execute actions, and interact with various systems.[1]

MCP follows a client-server architecture, defining how data (resources), interactive templates (prompts), and actionable functions (tools) are exposed by an MCP server and consumed by an MCP client (which could be an LLM host application or an AI agent).[1]

This lab covers two primary integration patterns:[1]

- Using existing MCP Servers within ADK: An ADK agent acts as an MCP client, leveraging tools provided by external MCP servers.
- Exposing ADK Tools via an MCP Server: Building an MCP server that wraps ADK tools, making them accessible to any MCP client.

## Objectives[1]

In this lab, you learn how to:[1]

- Use an ADK agent as an MCP client to interact with tools from existing MCP servers.
- Configure and deploy your own MCP server to expose ADK tools to other clients.
- Connect ADK agents with external tools through standardized MCP communication.
- Enable seamless interaction between LLMs and tools using Model Context Protocol.

## Task 1. Install ADK and set up your environment[1]

In this lab environment, the **Vertex AI API**, **Routes API** and **Directions API** have been enabled for you.[1]

### Prepare a Cloud Shell Editor tab[1]

1. With your Google Cloud console window selected, open Cloud Shell by pressing the **G** key and then the **S** key on your keyboard. Alternatively, you can click the Activate Cloud Shell button (Activate Cloud Shell) in the upper right of the Cloud console.
2. Click **Continue**.
3. When prompted to authorize Cloud Shell, click **Authorize**.
4. In the upper right corner of the Cloud Shell Terminal panel, click the **Open in new window** button Open in new window button.
5. In the Cloud Shell Terminal, enter the following to open the Cloud Shell Editor to your home directory:

   ```text
   cloudshell  
   workspace ~
   ```

6. Close any additional tutorial or Gemini panels that appear on the right side of the screen to save more of your window for your code editor.
7. Throughout the rest of this lab, you can work in this window as your IDE with the Cloud Shell Editor and Cloud Shell Terminal.

### Download and install ADK and code samples for this lab[1]

1. **Install ADK** by running the following command in the Cloud Shell Terminal.  

   **Note:** You will specify the version to ensure that the version of ADK that you install corresponds to the version used in this lab. You can view the latest version number and release notes at the [adk-python repo](https://github.com/google/adk-python/releases).

   ```bash
   sudo python3 -m pip install google-adk==1.5.0
   ```

   Copy

2. Paste the following commands into the Cloud Shell Terminal to copy a file from a Cloud Storage bucket, and unzip it, creating a project directory with code for this lab:

   ```bash
   gcloud storage cp gs://YOUR_GCP_PROJECT_ID-bucket/adk_mcp_tools.zip .
   unzip adk_mcp_tools.zip
   ```

   Copy

3. Install additional lab requirements with:

   ```bash
   python3 -m pip  
   install  
   -r adk_mcp_tools/requirements.txt
   ```

## Task 2. Using Google Maps MCP server with ADK agents (ADK as an MCP client) in adk web[1]

This section demonstrates how to integrate tools from an external Google Maps MCP server into your ADK agents. This is the most common integration pattern when your ADK agent needs to use capabilities provided by an existing service that exposes an MCP interface. You will see how the `MCPToolset` class can be directly added to your agent's `tools` list, enabling seamless connection to an MCP server, discovery of its tools, and making them available for your agent to use. These examples primarily focus on interactions within the `adk web` development environment.[1]

### MCPToolset[1]

The `MCPToolset` class is ADK's primary mechanism for integrating tools from an MCP server. When you include an `MCPToolset` instance in your agent's `tools` list, it automatically handles the interaction with the specified MCP server. Here's how it works:[1]

- **Connection Management**: On initialization, `MCPToolset` establishes and manages the connection to the MCP server. This can be a local server process (using `StdioServerParameters` for communication over standard input/output) or a remote server (using `SseServerParams` for Server-Sent Events). The toolset also handles the graceful shutdown of this connection when the agent or application terminates.
- **Tool Discovery & Adaptation**: Once connected, `MCPToolset` queries the MCP server for its available tools (via the `list_tools` MCP method). It then converts the schemas of these discovered MCP tools into ADK-compatible `BaseTool` instances.
- **Exposure to Agent**: These adapted tools are then made available to your `LlmAgent` as if they were native ADK tools.
- **Proxying Tool Calls**: When your `LlmAgent` decides to use one of these tools, `MCPToolset` transparently proxies the call (using the `call_tool` MCP method) to the MCP server, sends the necessary arguments, and returns the server's response back to the agent.
- **Filtering (Optional)**: You can use the `tool_filter` parameter when creating an MCPToolset to select a specific subset of tools from the MCP server, rather than exposing all of them to your agent.

### Get API key and Enable APIs[1]

In this sub-section, you will generate a new API key named **GOOGLE_MAPS_API_KEY**.[1]

1. **Open the browser tab displaying the Google Cloud Console** (not your Cloud Shell Editor).
2. You can **close the Cloud Shell Terminal pane** on this browser tab for more console area.
3. Search for **Credentials** in the search bar at the top of the page. Select it from the results.
4. On the **Credentials** page, click **+ Create Credentials** at the top of the page, then select **API key**.  

   The **API key created** dialog will display your newly created API key. Be sure to save this key locally for later use in the lab.

5. Click **Close** on the dialog box.  

   Your newly created key will be named **API Key 1** by default. Select the key, rename it to **GOOGLE_MAPS_API_KEY**, and click **Save**.

   Google Map Key

### Define your Agent with an MCPToolset for Google Maps[1]

In this sub-section, you will configure your agent to use the `MCPToolset` for Google Maps, enabling it to seamlessly provide directions and location-based information.[1]

1. In the Cloud Shell Editor's file explorer pane, find the **adk_mcp_tools** folder. Click it to toggle it open.
2. Navigate to the directory **adk_mcp_tools/google_maps_mcp_agent**.
3. Paste the following command in a plain text file, then update the `YOUR_ACTUAL_API_KEY` value with the Google Maps API key you generated and saved in a previous step:

   ```bash
   cd  
   ~/adk_mcp_tools
   cat <<  
   EOF > google_maps_mcp_agent/.env
   GOOGLE_GENAI_USE_VERTEXAI=TRUE
   GOOGLE_CLOUD_PROJECT=Project
   GOOGLE_CLOUD_LOCATION=Region
   GOOGLE_MAPS_API_KEY="YOUR_ACTUAL_API_KEY"
   MODEL=gemini_flash_model_id
   EOF
   ```

4. Copy and paste the updated command to Cloud Shell Terminal to run it and write a **.env** file which will provide authentication details for this agent directory.
5. Copy the **.env** file to the other agent directory you will use in this lab by running the following command:

   ```bash
   cp google_maps_mcp_agent/.env adk_mcp_server/.env
   ```

6. Next, add the following code where indicated in the `agent.py` file to add the Google maps tool to your agent. This will allow your agent to use the **MCPToolset** for Google Maps to provide directions or location-based information.

   ```python
   tools=[
       MCPToolset(
           connection_params=StdioConnectionParams(
               server_params=StdioServerParameters(
                   command='npx',
                   args=[
                       "-y",
                       "@modelcontextprotocol/server-google-maps",
                   ],
                   env={
                       "GOOGLE_MAPS_API_KEY": google_maps_api_key
                   }
               ),
               timeout=15,
           ),
       )
   ],
   ```


7. From the **adk_mcp_tools** project directory, launch the **Agent Development Kit Dev UI** with the following command:

   ```bash
   adk web
   ```

   **Output:**

   ```text
   INFO: Started server process [2434]
   INFO: Waiting for application startup.
   +----------------------------------------------------+
   | ADK Web Server started                             |
   |                                                    |
   | For local testing, access at http://localhost:8000.|
   +----------------------------------------------------+
   INFO: Application startup complete.
   INFO: Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
   ```

8. To view the web interface in a new tab, click the **http://127.0.0.1:8000** link in the Terminal output.
9. A new browser tab will open with the ADK Dev UI. From the **Select an agent** dropdown on the left, select the **google_maps_mcp_agent** from the dropdown.
10. Start a conversation with the agent and run the following prompts:

    ```text
    Get directions from GooglePlex to SFO.
    ``` 

    **Note:** If your API call times out the first time you use it, click **+ New Session** in the upper right of the ADK Dev UI and try again.

    ```text
    What's the route from Paris, France to Berlin, Germany?
    ```

    **Output:**

    Agent Response

11. **Click the agent icon** next to the agent's chat bubble with a lightning bolt, which indicates a function call. This will open up the Event inspector for this event:

    ADK Tool Call

12. **Notice that agent graph indicates several different tools**, identified by the wrench emoji (🔧). Even though you only imported one `MCPToolset`, that tool set came with the different tools you see listed here, such as `maps_place_details` and `maps_directions`.  

    The agent graph indicates several tools
13. On the **Request** tab, you can see the structure of the request. You can use the arrows at the top of the Event inspector to browse the agent's thoughts, function calls, and responses.
14. When you are finished asking questions of this agent, close the dev UI browser tab.
15. Go back to the Cloud Shell Terminal panel and press **CTRL + C** to stop the server.

## Task 3. Building an MCP server with ADK tools (MCP server exposing ADK)[1]

In this section, you'll learn how to expose the ADK `load_web_page` tool through a custom-built MCP server. This pattern allows you to wrap existing ADK tools and make them accessible to any standard MCP client application.[1]

### Create the MCP Server Script and Implement Server Logic[1]

1. Return to your Cloud Shell Editor tab and select the **adk_mcp_tools/adk_mcp_server** directory.
2. A Python file named `adk_server.py` has been prepared and commented for you. **Take some time to review that file**, reading the comments to understand how the code wraps a tool and serves it as an MCP server. Notice how it allows MCP clients to list available tools as well as invoke the ADK tool asynchronously, handling requests and responses in an MCP-compliant format.

### Test the Custom MCP Server with an ADK Agent[1]

3. Click on the **agent.py** file in the **adk_mcp_server** directory.
4. Update the path to your **adk_server.py** file.

   ```text
   /home/  
   Username  
   /adk_mcp_tools/  
   adk_mcp_server/adk_server.py
   ```

   Copy

5. Next, add the following code where indicated in the `agent.py` file to add the **MCPToolset** to your agent. An ADK agent acts as a client to the MCP server. This ADK agent will use `MCPToolset` to connect to your `adk_server.py` script.

   ```python
   tools=[
       MCPToolset(
           connection_params=StdioConnectionParams(
               server_params=StdioServerParameters(
                   command="python3",  # Command to run your MCP server script
                   args=[PATH_TO_YOUR_MCP_SERVER_SCRIPT],  # Argument is the path to the script
               ),
               timeout=15,
           ),
           tool_filter=[
               'load_web_page'
           ]  # Optional: ensure only specific tools are loaded
       )
   ],
   ```
6. To run the MCP server, start the `adk_server.py` script by running the following command in Cloud Shell Terminal:

   ```bash
   python3 ~/adk_mcp_tools/adk_mcp_server/adk_server.py
   ```

   **Output**:

  <img width="1456" height="130" alt="image" src="https://github.com/user-attachments/assets/37759c80-ac2f-4d87-ba81-3ac7114fa308" />


7. Open a new Cloud Shell Terminal tab by clicking the add-session-button button at the top of the Cloud Shell Terminal window.
8. In the Cloud Shell Terminal, from the **adk_mcp_tools** project directory, launch the **Agent Development Kit Dev UI** with the following command:

   ```bash
   cd  
   ~/adk_mcp_tools
   adk web
   ```

9. To view the web interface in a new tab, click the **http://127.0.0.1:8000** link in the Terminal output.
10. From the **Select an agent dropdown** on the left, select the **adk_mcp_server** from the dropdown.
11. Query the agent with:

    ```text
    Load the content from https://example.com.
    ```

    **Output**:

    <img width="3456" height="1900" alt="image" src="https://github.com/user-attachments/assets/01485c37-3ae7-460e-ba67-bcad588b7863" />


    What happens here:

    - The ADK agent (`web_reader_mcp_client_agent`) uses the `MCPToolset` to connect to your `adk_server.py`.
    - The MCP server will receive the `call_tool` request, execute the ADK `load_web_page` tool, and return the result.
    - The ADK agent will then relay this information. You should see logs from both the ADK Web UI (and its terminal) and from your `adk_server.py` terminal in the Cloud Shell Terminal tab where it is running.

    This demonstrates that ADK tools can be encapsulated within an MCP server, making them accessible to a broad range of MCP-compliant clients including ADK agents.

