
import random
import string
import json
from google.adk.events import Event
import plotly.graph_objects as go
from IPython.display import HTML, Markdown, display
import pandas as pd


def get_id(length: int = 8) -> str:
    """Generate a uuid of a specified length (default=8)."""
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def parse_adk_output_to_dictionary(events: list[Event], *, as_json: bool = False):
    """Parse ADK event output into a structured dictionary format,
    with the predicted trajectory dumped as a JSON string.

    """
    final_response = ""
    trajectory = []

    for event in events:
        if not getattr(event, "content", None) or not getattr(
            event.content, "parts", None
        ):
            continue
        for part in event.content.parts:
            if getattr(part, "function_call", None):
                info = {
                    "tool_name": part.function_call.name,
                    "tool_input": dict(part.function_call.args),
                }
                if info not in trajectory:
                    trajectory.append(info)
            if event.content.role == "model" and getattr(part, "text", None):
                final_response = part.text.strip()

    if as_json:
        trajectory_out = json.dumps(trajectory)
    else:
        trajectory_out = trajectory

    return {"response": final_response, "predicted_trajectory": trajectory_out}


def format_output_as_markdown(output: dict) -> str:
    """Convert the output dictionary to a formatted markdown string."""
    markdown = "### AI Response\n" + output["response"] + "\n\n"
    if output["predicted_trajectory"]:
        markdown += "### Function Calls\n"
        for call in output["predicted_trajectory"]:
            markdown += f"- **Function**: `{call['tool_name']}`\n"
            markdown += "  - **Arguments**\n"
            for key, value in call["tool_input"].items():
                markdown += f"    - `{key}`: `{value}`\n"
    return markdown


def display_eval_report(eval_result: pd.DataFrame) -> None:
    """Display the evaluation results."""
    display(Markdown("### Summary Metrics"))
    display(
        pd.DataFrame(eval_result.summary_metrics.items(), columns=["metric", "value"])
    )
    if getattr(eval_result, "metrics_table", None) is not None:
        display(Markdown("### Row‑wise Metrics"))
        display(eval_result.metrics_table.head())


def display_drilldown(row: pd.Series) -> None:
    """Displays a drill-down view for trajectory data within a row."""
    style = "white-space: pre-wrap; width: 800px; overflow-x: auto;"

    if not (
        isinstance(row["predicted_trajectory"], list)
        and isinstance(row["reference_trajectory"], list)
    ):
        return

    for predicted_trajectory, reference_trajectory in zip(
        row["predicted_trajectory"], row["reference_trajectory"], strict=False
    ):
        display(
            HTML(
                f"Tool Names:{predicted_trajectory['tool_name'], reference_trajectory['tool_name']}"
            )
        )

        if not (
            isinstance(predicted_trajectory.get("tool_input"), dict)
            and isinstance(reference_trajectory.get("tool_input"), dict)
        ):
            continue

        for tool_input_key in predicted_trajectory["tool_input"]:
            print("Tool Input Key: ", tool_input_key)

            if tool_input_key in reference_trajectory["tool_input"]:
                print(
                    "Tool Values: ",
                    predicted_trajectory["tool_input"][tool_input_key],
                    reference_trajectory["tool_input"][tool_input_key],
                )
            else:
                print(
                    "Tool Values: ",
                    predicted_trajectory["tool_input"][tool_input_key],
                    "N/A",
                )
        print("\n")
    display(HTML(""))


def display_dataframe_rows(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    num_rows: int = 3,
    display_drilldown: bool = False,
) -> None:
    """Displays a subset of rows from a DataFrame, optionally including a drill-down view."""
    if columns:
        df = df[columns]

    base_style = "font-family: monospace; font-size: 14px; white-space: pre-wrap; width: auto; overflow-x: auto;"
    header_style = base_style + "font-weight: bold;"

    for _, row in df.head(num_rows).iterrows():
        display(HTML("<h3>Example:</h3>"))
        for column in df.columns:
            display(
                HTML(
                    f"{column.replace('_', ' ').title()}: "
                )
            )
            display(HTML(f"{row[column]}"))

        display(HTML("<p><br /></p>"))

        if (
            display_drilldown
            and "predicted_trajectory" in df.columns
            and "reference_trajectory" in df.columns
        ):
            display_drilldown(row)


def plot_bar_plot(
    eval_result: pd.DataFrame, title: str, metrics: list[str] = None
) -> None:
    fig = go.Figure()
    data = []

    summary_metrics = eval_result.summary_metrics
    if metrics:
        summary_metrics = {
            k: v
            for k, v in summary_metrics.items()
            if any(selected_metric in k for selected_metric in metrics)
        }

    data.append(
        go.Bar(
            x=list(summary_metrics.keys()),
            y=list(summary_metrics.values()),
            name=title,
        )
    )

    fig = go.Figure(data=data)

    # Change the bar mode
    fig.update_layout(barmode="group")
    fig.show()


def display_radar_plot(eval_results, title: str, metrics=None):
    """Plot the radar plot."""
    fig = go.Figure()
    summary_metrics = eval_results.summary_metrics
    if metrics:
        summary_metrics = {
            k: v
            for k, v in summary_metrics.items()
            if any(selected_metric in k for selected_metric in metrics)
        }

    min_val = min(summary_metrics.values())
    max_val = max(summary_metrics.values())

    fig.add_trace(
        go.Scatterpolar(
            r=list(summary_metrics.values()),
            theta=list(summary_metrics.keys()),
            fill="toself",
            name=title,
        )
    )
    fig.update_layout(
        title=title,
        polar=dict(radialaxis=dict(visible=True, range=[min_val, max_val])),
        showlegend=True,
    )
    fig.show()