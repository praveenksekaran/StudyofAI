# Evaluate ADK agent performance using the Vertex AI Generative AI Evaluation Service[1]

## Overview[1]

Agent Development Kit (ADK) is a modular and extensible open-source framework for building AI agents. While ADK provides its own built-in evaluation module, this lab demonstrates how to use the **Vertex AI Generative AI Evaluation Service** to assess the performance of an ADK-based agent. This approach offers a broader, explainable, and quality-controlled toolkit to evaluate generative models or applications using custom metrics and human-aligned benchmarks.[1]

In this lab, you will walk through a step-by-step guide to evaluate your ADK agent using Vertex AI Gen AI Evaluation.[1]

This lab is based on [this notebook](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/evaluation/evaluating_adk_agent.ipynb) from the [GoogleCloudPlatform/generative-ai](https://github.com/GoogleCloudPlatform/generative-ai) repo.[1]

## Objective[1]

By the end of this lab, you will be able to:[1]

- Build and run a local ADK agent.
- Create and format an agent evaluation dataset.
- Evaluate agent performance using:
  -Single tool usage evaluation
  -Trajectory-based evaluation
  - Response quality evaluation
- Use Vertex AI’s Evaluation Service to generate explainable metrics and benchmark results.

## Task 1. Prepare the environment in Vertex AI Workbench[1]

1.In the Google Cloud console, search for the **Vertex AI Dashboard** at the top of the console and select the first result.  

2.Click on **Enable all recommended APIs**.  

3.In the left-hand Vertex AI navigation pane, under Notebooks click on **Workbench**.  

4.Under **Instances**, click on **Open JupyterLab** next to your `vertex-ai-jupyterlab` instance. JupyterLab will launch in a new tab.  

5.On the JupyterLab Launcher page, click **Terminal** to open a Shell.  

6.Download a notebook and helper file for this lab with the command:  

  ```text
  gcloud storage cp gs://YOUR_GCP_PROJECT_ID-bucket/* .
  ```

7.Open the `notebook name` file.  

8.In the **Select Kernel** dialog, choose **Python 3** from the list of available kernels.  

9.Run the first code cell under the header **Install Agent Development Kit (ADK) and other required packages** to install dependencies. To run a cell, click the play button at the top or highlight the cell and press SHIFT+ENTER on your keyboard.  

10.To use the newly installed packages in this Jupyter runtime, you must restart the runtime. Wait for the `[*]` beside the cell to change to `[1]` to show that the cell has completed, then in the Jupyter Lab menus, select **Kernel > Restart Kernel and Clear All Outputs...**.  

11.When prompted to confirm, select **Restart**.  

12.Once the kernel has restarted, run the next cell under **Set Google Cloud project information** to set Google Cloud project information and initialize **Vertex AI**.  

13.Run the next cell in the **Import libraries** section of the notebook.  

## Task 2. Read and run the rest of the notebook[1]

Read and run through the cells of the notebook. In its sections, you will:[1]

- **Set tools** to define `get_product_details()` and `get_product_price()` functions that your agent will later use as tools.
- **Set the Model** to define the Gemini model your agent will use.
- **Assemble the agent** by defining a function that builds an Agent, providing it the tools you've defined above, and uses it to generate a response to a query.
- **Test the Agent** to see a couple of example responses, including Markdown-formatted output citing which functions were called.
- Learn about **Evaluating an ADK agent with Vertex AI Gen AI Evaluation**.
- **Prepare an Agent Evaluation Dataset** consisting of a set of prompts and their expected tool calls (reference trajectory).
- Run **Single tool usage evaluation** to determine if the agent is choosing the correct single tool for a given task.
- Run **Trajectory Evaluation** to assess whether the agent not only chooses the right tools but also utilizes them in your intended order.
- Learn to **Visualize Evaluation Results**.
- Learn to **Define a custom metric** and evaluate on it.
- A bonus **Bring-Your-Own-Dataset (BYOD)** section shows you how to evaluate agent responses generated elsewhere by providing an evaluation dataset containing those responses.


[1](https://www.skills.google/paths/3273/course_templates/1275/labs/606602)
