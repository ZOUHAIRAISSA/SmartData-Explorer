import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from sklearn.preprocessing import LabelEncoder
import numpy as np

class DataVisualizerClass(tk.Tk):
    def __init__(self):
        super().__init__()

    @staticmethod
    def handle_close( window):
        window.destroy()
    
    @staticmethod
    def show_histograms_view(self, dfP):
        if dfP is None or dfP.empty:
            messagebox.showwarning("Warning", "No dataset loaded!")
            return

        # Create a new window for histograms
        histograms_window = tk.Toplevel(self)
        histograms_window.title("Histograms")

        # Canvas and scrollbar for histograms
        canvas = tk.Canvas(histograms_window)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        histograms_window.protocol("WM_DELETE_WINDOW", lambda: DataVisualizerClass.handle_close(histograms_window))

        scrollbar = ttk.Scrollbar(histograms_window, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Frame inside canvas to contain histograms
        histograms_frame = ttk.Frame(canvas)
        histograms_frame.pack(fill=tk.BOTH, expand=True)
        canvas.create_window((0, 0), window=histograms_frame, anchor=tk.NW)

        # Preprocess data
        self.df = dfP

        # Create histograms for each column
        num_columns = len(self.df.columns)
        for i, col in enumerate(self.df.columns):
            if i % 2 == 0:
                # Create a new row frame for every two histograms
                row_frame = ttk.Frame(histograms_frame)
                row_frame.pack(fill=tk.BOTH, padx=10, pady=10)

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(self.df[col], kde=True, color='blue', ax=ax)
            ax.set_title(f'Histogram of {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            ax.grid(True)

            # Embedding matplotlib plot into tkinter
            canvas_plot = FigureCanvasTkAgg(fig, master=row_frame)
            canvas_plot.draw()
            canvas_plot.get_tk_widget().pack(side=tk.LEFT, padx=10, pady=10)

        # Configure scrollbar and canvas
        histograms_frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))
        canvas.config(yscrollcommand=scrollbar.set)

    @staticmethod
    def show_heatmap_view(self, dfP):
        if dfP is None or dfP.empty:
            messagebox.showwarning("Warning", "No dataset loaded!")
            return

        # Create a new window for heatmap
        heatmap_window = tk.Toplevel(self)
        heatmap_window.title("Heatmap")
        heatmap_window.protocol("WM_DELETE_WINDOW", lambda: DataVisualizerClass.handle_close(heatmap_window))

        # Frame to contain the heatmap plot
        heatmap_frame = ttk.Frame(heatmap_window)
        heatmap_frame.pack(fill=tk.BOTH, expand=True)

        # Preprocess data
        self.df = dfP

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(self.df.corr(), annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Correlation Heatmap')
        ax.set_xlabel('Features')
        ax.set_ylabel('Features')

        # Embedding matplotlib plot into tkinter
        canvas = FigureCanvasTkAgg(fig, master=heatmap_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    @staticmethod
    def show_pie_chart_view(self):
        if self.df is None or self.df.empty:
            messagebox.showwarning("Warning", "No dataset loaded!")
            return

        # Find all categorical columns
        categorical_columns = self.df.select_dtypes(include=['object']).columns

        if len(categorical_columns) == 0:
            messagebox.showwarning("Warning", "No categorical columns available!")
            return

        # Create a new window for Pie Charts
        pie_charts_window = tk.Toplevel(self)
        pie_charts_window.title("Pie Charts")
        pie_charts_window.protocol("WM_DELETE_WINDOW", lambda: DataVisualizerClass.handle_close(pie_charts_window))

        # Iterate over each categorical column
        for col in categorical_columns:
            # Frame to contain the Pie Chart plot for each column
            pie_chart_frame = ttk.Frame(pie_charts_window)
            pie_chart_frame.pack(fill=tk.BOTH, expand=True)

            counts = self.df[col].value_counts()

            # Plotting Pie Chart
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
            ax.set_title(f'Pie Chart of {col}')

            # Embedding matplotlib plot into tkinter
            canvas_plot = FigureCanvasTkAgg(fig, master=pie_chart_frame)
            canvas_plot.draw()
            canvas_plot.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    @staticmethod
    def show_violin_plots_view(self, dfP):
        if dfP is None or dfP.empty:
            messagebox.showwarning("Warning", "No dataset loaded!")
            return

        # Create a new window for violin plots
        violin_plots_window = tk.Toplevel(self)
        violin_plots_window.title("Violin Plots")

        # Canvas and scrollbar for displaying plots
        canvas = tk.Canvas(violin_plots_window)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        violin_plots_window.protocol("WM_DELETE_WINDOW", lambda: DataVisualizerClass.handle_close(violin_plots_window))

        scrollbar = ttk.Scrollbar(violin_plots_window, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Frame inside canvas to contain plots
        plot_container = ttk.Frame(canvas)
        plot_container.pack(fill=tk.BOTH, expand=True)
        canvas.create_window((0, 0), window=plot_container, anchor=tk.NW)

        # Preprocess data (fill missing values if any)
        df_processed = dfP

        # Create violin plots for each numeric column
        num_columns = df_processed.select_dtypes(include=[np.number]).columns
        num_plots = len(num_columns)

        # Define the layout of plots (e.g., 2 plots per row)
        num_plots_per_row = 2
        num_rows = (num_plots + num_plots_per_row - 1) // num_plots_per_row  # Calculate number of rows

        for i, col in enumerate(num_columns):
            row = i // num_plots_per_row
            column = i % num_plots_per_row

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.violinplot(x=df_processed[col], ax=ax)
            ax.set_title(f'Violin Plot of {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Value')

            # Embedding matplotlib plot into tkinter
            canvas_plot = FigureCanvasTkAgg(fig, master=plot_container)
            canvas_plot.draw()
            canvas_plot.get_tk_widget().grid(row=row, column=column, padx=10, pady=10)

        # Update canvas view
        plot_container.update_idletasks()
        canvas.config(scrollregion=canvas.bbox('all'))

        # Adding toolbar for navigation
        toolbar = NavigationToolbar2Tk(canvas_plot, plot_container)
        toolbar.update()
        toolbar.grid(row=num_rows, columnspan=num_plots_per_row, sticky='ew')

    @staticmethod
    def show_bar_charts_view(self, dfP):
        if dfP is None or dfP.empty:
            messagebox.showwarning("Warning", "No dataset loaded!")
            return

        # Create a new window for bar charts
        bar_charts_window = tk.Toplevel(self)
        bar_charts_window.title("Bar Charts")
        bar_charts_window.protocol("WM_DELETE_WINDOW", lambda: DataVisualizerClass.handle_close(bar_charts_window))

        # Canvas and scrollbar for bar charts
        canvas = tk.Canvas(bar_charts_window)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(bar_charts_window, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Frame inside canvas to contain bar charts
        bar_charts_frame = ttk.Frame(canvas)
        bar_charts_frame.pack(fill=tk.BOTH, expand=True)
        canvas.create_window((0, 0), window=bar_charts_frame, anchor=tk.NW)

        # Preprocess data (if needed)
        df_processed = dfP

        # Create bar charts for each column
        num_columns = len(df_processed.columns)
        for i, col in enumerate(df_processed.columns):
            if i % 2 == 0:
                # Create a new row frame for every two bar charts
                row_frame = ttk.Frame(bar_charts_frame)
                row_frame.pack(fill=tk.BOTH, padx=10, pady=10)

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.countplot(x=col, data=df_processed, ax=ax)
            ax.set_title(f'Bar Chart of {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Count')
            ax.grid(True)

            # Embedding matplotlib plot into tkinter
            canvas = FigureCanvasTkAgg(fig, master=row_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.LEFT, padx=10, pady=10)

        # Configure scrollbar and canvas
        bar_charts_frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))
        canvas.config(yscrollcommand=scrollbar.set)

    @staticmethod
    def show_box_plots_view(self, dfP):
        if dfP is None or dfP.empty:
            messagebox.showwarning("Warning", "No dataset loaded!")
            return

        # Create a new window for box plots
        box_plots_window = tk.Toplevel(self)
        box_plots_window.title("Box Plots")
        box_plots_window.protocol("WM_DELETE_WINDOW", lambda: DataVisualizerClass.handle_close(box_plots_window))

        # Canvas and scrollbar for displaying plots
        canvas = tk.Canvas(box_plots_window)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(box_plots_window, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Frame inside canvas to contain plots
        plot_container = ttk.Frame(canvas)
        plot_container.pack(fill=tk.BOTH, expand=True)
        canvas.create_window((0, 0), window=plot_container, anchor=tk.NW)

        # Preprocess data (fill missing values if any)
        df_processed = dfP

        # Create box plots for each numeric column
        num_columns = df_processed.select_dtypes(include=[np.number]).columns
        num_plots = len(num_columns)

        # Define the layout of plots (e.g., 2 plots per row)
        num_plots_per_row = 2
        num_rows = (num_plots + num_plots_per_row - 1) // num_plots_per_row  # Calculate number of rows

        for i, col in enumerate(num_columns):
            row = i // num_plots_per_row
            column = i % num_plots_per_row

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(y=df_processed[col], ax=ax)
            ax.set_title(f'Box Plot of {col}')
            ax.set_ylabel(col)

            # Embedding matplotlib plot into tkinter
            canvas_plot = FigureCanvasTkAgg(fig, master=plot_container)
            canvas_plot.draw()
            canvas_plot.get_tk_widget().grid(row=row, column=column, padx=10, pady=10)

        # Update canvas view
        plot_container.update_idletasks()
        canvas.config(scrollregion=canvas.bbox('all'))

        # Adding toolbar for navigation
        toolbar = NavigationToolbar2Tk(canvas_plot, plot_container)
        toolbar.update()
        toolbar.grid(row=num_rows, columnspan=num_plots_per_row, sticky='ew')

    @staticmethod
    def show_pair_plot_view(self, dfP):
        if dfP is None or dfP.empty:
            messagebox.showwarning("Warning", "No dataset loaded!")
            return

        # Create a new window for pair plot
        pair_plot_window = tk.Toplevel(self)
        pair_plot_window.title("Pair Plot")
        pair_plot_window.protocol("WM_DELETE_WINDOW", lambda: DataVisualizerClass.handle_close(pair_plot_window))

        # Canvas and scrollbars for displaying the plot
        canvas = tk.Canvas(pair_plot_window)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        vertical_scrollbar = ttk.Scrollbar(pair_plot_window, orient=tk.VERTICAL, command=canvas.yview)
        vertical_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        horizontal_scrollbar = ttk.Scrollbar(pair_plot_window, orient=tk.HORIZONTAL, command=canvas.xview)
        horizontal_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Frame inside canvas to contain the plot
        plot_container = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=plot_container, anchor=tk.NW)

        # Preprocess data (fill missing values if any)
        df_processed = dfP

        # Create the pair plot
        pair_plot_fig = sns.pairplot(df_processed)

        # Embedding matplotlib plot into tkinter
        canvas_plot = FigureCanvasTkAgg(pair_plot_fig.fig, master=plot_container)
        canvas_plot.draw()
        canvas_plot.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Update canvas view
        plot_container.update_idletasks()
        canvas.config(scrollregion=canvas.bbox('all'))
        canvas.config(yscrollcommand=vertical_scrollbar.set)
        canvas.config(xscrollcommand=horizontal_scrollbar.set)

        # Adding toolbar for navigation
        toolbar = NavigationToolbar2Tk(canvas_plot, plot_container)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)

    @staticmethod
    def show_scatter_plots_view(self, dfP):
        if dfP is None or dfP.empty:
            messagebox.showwarning("Warning", "No dataset loaded!")
            return

        # Create a new window for scatter plots
        scatter_plots_window = tk.Toplevel(self)
        scatter_plots_window.title("Scatter Plots")

        # Canvas and scrollbar for displaying plots
        canvas = tk.Canvas(scatter_plots_window)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(scatter_plots_window, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Frame inside canvas to contain plots
        plot_container = ttk.Frame(canvas)
        plot_container.pack(fill=tk.BOTH, expand=True)
        canvas.create_window((0, 0), window=plot_container, anchor=tk.NW)

        # Preprocess data (fill missing values if any)
        df_processed = dfP

        # Create scatter plots for each pair of numeric columns
        num_columns = df_processed.select_dtypes(include=[np.number]).columns
        num_plots = len(num_columns) * (len(num_columns) - 1) // 2  # Number of pairs

        # Define the layout of plots (e.g., 2 plots per row)
        num_plots_per_row = 2
        num_rows = (num_plots + num_plots_per_row - 1) // num_plots_per_row  # Calculate number of rows

        plot_index = 0
        for i, col1 in enumerate(num_columns):
            for j, col2 in enumerate(num_columns):
                if i < j:  # Avoid duplicate pairs and self-pairs
                    row = plot_index // num_plots_per_row
                    column = plot_index % num_plots_per_row

                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.scatterplot(x=df_processed[col1], y=df_processed[col2], ax=ax)
                    ax.set_title(f'Scatter Plot of {col1} vs {col2}')
                    ax.set_xlabel(col1)
                    ax.set_ylabel(col2)

                    # Embedding matplotlib plot into tkinter
                    canvas_plot = FigureCanvasTkAgg(fig, master=plot_container)
                    canvas_plot.draw()
                    canvas_plot.get_tk_widget().grid(row=row, column=column, padx=10, pady=10)

                    plot_index += 1

        # Update canvas view
        plot_container.update_idletasks()
        canvas.config(scrollregion=canvas.bbox('all'))

        # Adding toolbar for navigation
        toolbar = NavigationToolbar2Tk(canvas_plot, plot_container)
        toolbar.update()
        toolbar.grid(row=num_rows, columnspan=num_plots_per_row, sticky='ew')

    @staticmethod
    def show_line_plot_view(self, dfP):
        if dfP is None or dfP.empty:
            messagebox.showwarning("Warning", "No dataset loaded!")
            return

        # Create a new window for line plots
        line_plots_window = tk.Toplevel(self)
        line_plots_window.title("Line Plots")

        # Canvas and scrollbars for displaying the plots
        canvas = tk.Canvas(line_plots_window)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        vertical_scrollbar = ttk.Scrollbar(line_plots_window, orient=tk.VERTICAL, command=canvas.yview)
        vertical_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        horizontal_scrollbar = ttk.Scrollbar(line_plots_window, orient=tk.HORIZONTAL, command=canvas.xview)
        horizontal_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Frame inside canvas to contain the plots
        plot_container = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=plot_container, anchor=tk.NW)

        # Preprocess data (fill missing values if any)
        df_processed = dfP

        # Create line plots for each numeric column
        num_columns = df_processed.select_dtypes(include=[np.number]).columns
        num_plots = len(num_columns)

        # Define the layout of plots (e.g., 2 plots per row)
        num_plots_per_row = 2
        num_rows = (num_plots + num_plots_per_row - 1) // num_plots_per_row  # Calculate number of rows

        for i, col in enumerate(num_columns):
            row = i // num_plots_per_row
            col_position = i % num_plots_per_row

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(df_processed.index, df_processed[col], marker='o', linestyle='-', color='b')
            ax.set_title(f'Line Plot of {col}')
            ax.set_xlabel('Index')
            ax.set_ylabel(col)
            ax.grid(True)

            # Embedding matplotlib plot into tkinter
            canvas_plot = FigureCanvasTkAgg(fig, master=plot_container)
            canvas_plot.draw()
            canvas_plot.get_tk_widget().grid(row=row, column=col_position, padx=10, pady=10)

        # Update canvas view
        plot_container.update_idletasks()
        canvas.config(scrollregion=canvas.bbox('all'))
        canvas.config(yscrollcommand=vertical_scrollbar.set)
        canvas.config(xscrollcommand=horizontal_scrollbar.set)

        # Adding toolbar for navigation
        toolbar = NavigationToolbar2Tk(canvas_plot, plot_container)
        toolbar.update()
        toolbar.grid(row=num_rows, columnspan=num_plots_per_row, sticky='ew')

    @staticmethod
    def show_general_info(self):
        if self.df is None or self.df.empty:
            messagebox.showwarning("Warning", "No dataset loaded!")
            return

        # Create a new window for general info
        general_info_window = tk.Toplevel(self)
        general_info_window.title("General Information")

        # Frame inside window to contain general info
        info_frame = ttk.Frame(general_info_window)
        info_frame.pack(fill=tk.BOTH, expand=True)

        # General Info Text
        info_text = tk.Text(info_frame, wrap=tk.WORD)
        info_text.pack(fill=tk.BOTH, expand=True)

        # Get general information about the dataset
        buffer = []
        buffer.append(f"Dataset Shape: {self.df.shape}")
        buffer.append("\n\nColumns and Data Types:\n")
        buffer.append(str(self.df.dtypes))
        buffer.append("\n\nMissing Values:\n")
        buffer.append(str(self.df.isnull().sum()))
        buffer.append("\n\nStatistical Summary:\n")
        buffer.append(str(self.df.describe(include='all')))
        
        # Write general info to the text widget
        info_text.insert(tk.END, "\n".join(buffer))

        # Make the text widget read-only
        info_text.config(state=tk.DISABLED)

