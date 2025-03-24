import matplotlib.pyplot as plt
import pandas as pd

def plot_bar_chart(data, x_col, y_col, y_err_col=None, title="", xlabel="", ylabel="", output_file=""):
    """
    Plot a bar chart with optional error bars.
    Args:
        data (pd.DataFrame): Data to plot.
        x_col (str): Column for x-axis.
        y_col (str): Column for y-axis.
        y_err_col (str): Column for error bars (optional).
        title (str): Chart title.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
        output_file (str): Path to save the chart.
    """
    plt.figure(figsize=(10, 6))
    if y_err_col:
        plt.bar(data[x_col], data[y_col], yerr=data[y_err_col])
    else:
        plt.bar(data[x_col], data[y_col])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file)
    plt.show()

def plot_line_chart(data, x_col, y_col, title="", xlabel="", ylabel="", output_file=""):
    """
    Plot a line chart.
    Args:
        data (pd.DataFrame): Data to plot.
        x_col (str): Column for x-axis.
        y_col (str): Column for y-axis.
        title (str): Chart title.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
        output_file (str): Path to save the chart.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(data[x_col], data[y_col], label=y_col)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file)
    plt.show()
