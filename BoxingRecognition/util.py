import numpy as np
import pandas as pd
from typing import List, Optional
import seaborn as sns
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from matplotlib import cm
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import plotly.graph_objects as go

from BoxingRecognition import dba


class Window:
    def __init__(self, window_id: int, max_window_size: int, sensor_type: str):
        """
        Initializes an empty Window object (x, y, z, labels)
        :param window_id: ID of the window
        :param max_window_size: Maximum size of the window (number of data points)
        :param sensor_type: Type of sensor (accelerometer, gyroscope, magnetometer)
        """
        # Store the parameters
        self.window_id = window_id
        self.max_window_size = max_window_size
        self.sensor_type = sensor_type

        # Initialize empty data lists
        self.x: List[float] = []
        self.y: List[float] = []
        self.z: List[float] = []
        self.labels: List[float] = []

    def __str__(self):
        return (f"Window (ID: {self.window_id}) (Sensor Type: {self.sensor_type}) (Current Size: {len(self)})"
                f"(Max Size: {self.max_window_size}) (Most frequent label: {self.get_label()})")

    def __len__(self):
        return len(self.x)

    def is_full(self) -> bool:
        """
        Returns True if the window is full, False otherwise
        """
        return len(self) == self.max_window_size

    def add_single_data_point(self, x: float, y: float, z: float, label: float | None) -> None:
        """
        Adds data to the window (single data point)
        :param x: X-axis data
        :param y: Y-axis data
        :param z: Z-axis data
        :param label: Label for the data (or None if the data is not labeled)
        :return: None
        """
        # Check if the window is full
        if self.is_full():
            raise ValueError("Window is full, cannot add more data")

        # Check if all the data is of the correct type
        if not all(isinstance(v, float) for v in [x, y, z]):
            raise ValueError(
                f"All data must be floats, received types: x={type(x)}, y={type(y)}, z={type(z)}")
        if label is not None and not isinstance(label, float):
            raise ValueError(f"Label must be a float or None, received type: {type(label)}")

        # Data is valid, window is not full, add the data
        self.x.append(x)
        self.y.append(y)
        self.z.append(z)
        self.labels.append(label)

    def add_multiple_data_points(self, x: List[float], y: List[float], z: List[float], labels: List[float]) -> None:
        """
        Adds data to the window (multiple data points)
        :param x: X-axis data
        :param y: Y-axis data
        :param z: Z-axis data
        :param labels: Labels for the data
        :return: None
        """
        # Check if the window is full
        if self.is_full():
            raise ValueError("Window is full, cannot add more data")

        # Helper function to check if all elements in a list are of type float
        def check_float_list(lst):
            return isinstance(lst, list) and all(isinstance(i, float) for i in lst)

        # Check if x, y, z
        if not (check_float_list(x) and check_float_list(y) and check_float_list(z)):
            raise ValueError("All data must be lists of floats")
        # Labels can be a list of None's as well, next to floats
        if not (check_float_list(labels) or all(i is None for i in labels)):
            raise ValueError("Labels must be a list of floats or None's")

        # Check if adding the data will exceed the window size
        if len(self) + len(x) > self.max_window_size:
            raise ValueError(
                f"Adding this data will exceed the window size (current size: {len(self)}, after adding:"
                f"{len(self) + len(x)}), limit is {self.max_window_size}")

        # Data is valid, window is not full, will not exceed the window, add the data
        self.x.extend(x)
        self.y.extend(y)
        self.z.extend(z)
        self.labels.extend(labels)

    def extract_features(self) -> pd.DataFrame:
        features = {}

        # Process each axis separately
        for axis_name, axis_data in zip(['x', 'y', 'z'], [self.x, self.y, self.z]):
            axis_data = np.array(axis_data)

            # Only gyroscope y-axis, and acceleromet_x data is used for feature extraction
            if axis_name != 'x' and self.sensor_type == 'accelerometer':
                continue
            if axis_name != 'y' and self.sensor_type == 'gyroscope':
                continue

            # Calculate features
            min = np.min(axis_data)
            max = np.max(axis_data)
            std = np.std(axis_data)
            var = np.var(axis_data)
            rms = np.sqrt(np.mean(np.square(axis_data)))
            quantile_10 = np.quantile(axis_data, 0.1)
            abs_max = np.max(np.abs(axis_data))

            # Update features
            features.update({
                f"{self.sensor_type}_{axis_name}_min": min,
                f"{self.sensor_type}_{axis_name}_max": max,
                f"{self.sensor_type}_{axis_name}_std": std,
                f"{self.sensor_type}_{axis_name}_var": var,
                f"{self.sensor_type}_{axis_name}_rms": rms,
                f"{self.sensor_type}_{axis_name}_quantile_10": quantile_10,
                f"{self.sensor_type}_{axis_name}_abs_max": abs_max
            })

        # Return a pandas DataFrame with the features as columns and a single row of values
        return pd.DataFrame([features])

    def get_label(self) -> float:
        """
        Returns the most frequent label in the window, or -1 if there are no labels
        """
        if not self.labels:
            return -1
        return max(set(self.labels), key=self.labels.count)


class SlidingWindow:
    def __init__(self, sensor_type: str, window_size: int, overlap_size: int = 0):
        """
        A wrapper class for the Window class that generates windows from a stream of data
        :param sensor_type: Type of sensor (accelerometer, gyroscope)
        :param window_size: Size of windows to generate from the data (number of data points in each window)
        :param overlap_size: Number of data points to overlap between windows
        """
        # Store the parameters
        self.sensor_type = sensor_type
        self.window_size = window_size
        self.overlap_size = overlap_size

        # Initialize an empty window
        self.current_window = Window(window_id=0, max_window_size=window_size, sensor_type=sensor_type)

    def add_data(self, x: float, y: float, z: float, label: float | None) -> Optional[Window]:
        """
        Processes data, returns a Window object if the window is full
        :param x: X-axis data
        :param y: Y-axis data
        :param z: Z-axis data
        :param label: Label for the data
        :return: A Window object if the window is full, None otherwise
        """

        # Append the data to the current window
        self.current_window.add_single_data_point(x, y, z, label)

        # Check if the current window is full (as a result of adding the data)
        if self.current_window.is_full():
            # Create a copy of the current window
            window = self.current_window

            # Create a new window
            self.current_window = Window(sensor_type=self.sensor_type,
                                         max_window_size=self.window_size, window_id=self.current_window.window_id + 1)

            # Carry over the overlap (if applicable)
            if self.overlap_size > 0:
                self.current_window.add_multiple_data_points(
                    window.x[-self.overlap_size:],
                    window.y[-self.overlap_size:],
                    window.z[-self.overlap_size:],
                    window.labels[-self.overlap_size:]
                )
            return window

        return None


class DataUtility:

    @staticmethod
    def get_aggregate_df(sessions: list, drop_unlabeled_data: bool = False,
                         base_path: str = "../Data/{}/labeled_{}.csv", augment_data: bool = False,
                         shuffle_sequences: bool = False) -> pd.DataFrame:
        """
        Loads labeled raw data from specified sessions and aggregates it into a single DataFrame, with a single label
        column.
        :param sessions: List of session numbers to include in the aggregation.
        :param drop_unlabeled_data: If True, rows without a label will be dropped (Normally used for idle data).
        :param base_path: The base path for the data files, with placeholders for session number and movement.
        :param augment_data: If True, data augmentation will be applied to the data.
        :return: A DataFrame containing labeled data from specified sessions, with a single label column (0-5).
        """
        # Define paths
        base_path = base_path
        movements = ['slip_left', 'slip_right', 'roll_left', 'roll_right', 'pull_back']

        def label_extractor(row):
            for label in label_columns:
                if row[label] == 'x':  # Labels are marked with 'x'
                    # Get column name without 'label_Boxing_' prefix
                    col_name = label.replace('label_Boxing_', '')
                    # Map the column names to numbers and ensure the output is an integer
                    if col_name == 'slip_left':
                        return 1
                    if col_name == 'slip_right':
                        return 2
                    if col_name == 'roll_left':
                        return 3
                    if col_name == 'roll_right':
                        return 4
                    if col_name == 'pullback':
                        return 5
                    else:
                        raise ValueError(f"Unknown label: {col_name}")
            # Return 0 for rows without a label
            return 0

        # Load data from all sessions into a single DataFrame
        df = pd.DataFrame()
        for session in sessions:
            session_df = pd.DataFrame()
            for movement in movements:
                file_path = base_path.format(session, movement)
                try:
                    session_df = pd.concat([session_df, pd.read_csv(file_path)], ignore_index=True)
                except FileNotFoundError:
                    print(f"Warning: {file_path} not found and will be skipped. (session {session})")
                except pd.errors.EmptyDataError:
                    print(f"Warning: {file_path} is empty and will be skipped. (session {session})")

            # Merge the gesture columns into a single label column
            label_columns = [col for col in session_df.columns if 'label' in col]
            session_df['label'] = session_df.apply(label_extractor, axis=1).astype(float)
            session_df = session_df.drop(columns=label_columns)

            # Negate the x-axis values of the sensors
            session_df['sensor_accX'] = -session_df['sensor_accX']
            session_df['sensor_gyroX'] = -session_df['sensor_gyroX']

            # If drop_unlabeled_data is True, remove rows where label is 0
            if drop_unlabeled_data:
                session_df = session_df[session_df['label'] != 0]

            df = pd.concat([df, session_df], ignore_index=True)

        # Augment the data if requested
        if augment_data:
            # Scaling: Different multipliers
            for multiplier in [2, 0.5]:
                df_augmented = df.copy()
                df_augmented[['sensor_accX', 'sensor_accY', 'sensor_accZ']] *= multiplier
                df_augmented[['sensor_gyroX', 'sensor_gyroY', 'sensor_gyroZ']] *= multiplier
                df = pd.concat([df, df_augmented], ignore_index=True)
                print(f"Data augmented: Multiplier: {multiplier}")

        # Shuffle data if requested
        if shuffle_sequences:
            blocks = []
            current_label = None
            start_index = 0
            for i, label in tqdm(enumerate(df['label']), total=len(df['label']), desc="Shuffling data"):
                if label != current_label:
                    blocks.append((start_index, i))
                    start_index = i
                    current_label = label
            blocks.append((start_index, len(df['label'])))
            np.random.shuffle(blocks)
            new_df = pd.DataFrame()
            for start, end in tqdm(blocks, total=len(blocks), desc="Shuffling data"):
                # Drop around 50% of the blocks with label 0
                if df.iloc[start]['label'] == 0 and np.random.rand() < 0.5:
                    continue
                new_df = pd.concat([new_df, df.iloc[start:end]], ignore_index=True)
            df = new_df

        # Return the dataframe containing all sessions
        return df

    @staticmethod
    def get_feature_df(aggregated_df, window_size, overlap_size, drop_window_id=False):
        """
        Converts a dataframe containing labeled raw data to a dataframe containing features (using windows)
        :param aggregated_df: The labeled DataFrame
        :param window_size: The size of the windows
        :param overlap_size: The overlap between windows
        :param drop_window_id: If True, the window ID column will be dropped
        :return: A DataFrame containing the features of the labeled data (using windows)
        """
        # Prepare a list to collect the windows
        accelerometer_windows = []
        gyroscope_windows = []

        # Prepare sliding windows to collect the data
        accelerometer_sliding_window = SlidingWindow("accelerometer", window_size, overlap_size)
        gyroscope_sliding_window = SlidingWindow("gyroscope", window_size, overlap_size)

        # Loop through the data
        for index, row in tqdm(aggregated_df.iterrows(), total=len(aggregated_df),
                               desc="Processing data (Creating windows)"):
            acc_x, acc_y, acc_z = row['sensor_accX'], row['sensor_accY'], row['sensor_accZ']
            gyro_x, gyro_y, gyro_z = row['sensor_gyroX'], row['sensor_gyroY'], row['sensor_gyroZ']
            label = row['label']

            # Add the data to the window generators
            accelerometer_window = accelerometer_sliding_window.add_data(acc_x, acc_y, acc_z, label)
            gyroscope_window = gyroscope_sliding_window.add_data(gyro_x, gyro_y, gyro_z, label)

            # Check if the windows were full (Got a Window object back), add them to the lists if they were
            if accelerometer_window is not None:
                accelerometer_windows.append(accelerometer_window)
            if gyroscope_window is not None:
                gyroscope_windows.append(gyroscope_window)

        # All windows are collected, extract features, and create a DataFrame
        rows = []
        for acc_window, gyro_window in tqdm(zip(accelerometer_windows, gyroscope_windows),
                                            total=len(accelerometer_windows),
                                            desc="Processing windows (Extracting features)"):
            # Extra information to create a row
            window_id = acc_window.window_id
            label = acc_window.get_label()
            acc_features = acc_window.extract_features()
            gyro_features = gyro_window.extract_features()

            # Create a row for the DataFrame
            row = {**acc_features.iloc[0].to_dict(), **gyro_features.iloc[0].to_dict(), 'label': label,
                   'Window ID': window_id}

            # Append the row to the list
            rows.append(row)

        df = pd.DataFrame(rows)

        # Show how many times a unique label appears in the dataset
        EvaluationUtility.plot_label_distribution(df)

        # Drop the window ID column if requested
        if drop_window_id:
            return df.drop(columns='Window ID')

        return df

    @staticmethod
    def collect_label_dataframes(df: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Collects sequences of rows with the same label from a DataFrame
        :param df: The DataFrame to collect sequences from
        :return: Lists of DataFrames, with each list containing DataFrames for a specific label
        """

        label_dataframes = {}
        current_label = None
        sequence_rows = []

        for index, row in df.iterrows():
            label = row['label']
            if label != current_label:
                if sequence_rows:  # Check if there are collected rows to be added
                    if current_label not in label_dataframes:
                        label_dataframes[current_label] = []
                    label_dataframes[current_label].append(pd.DataFrame(sequence_rows))
                    sequence_rows = []  # Reset the list for new label rows

                current_label = label

            sequence_rows.append(row.to_dict())  # Append current row as a dictionary

        # Handle the last sequence in the DataFrame
        if sequence_rows and current_label:
            if current_label not in label_dataframes:
                label_dataframes[current_label] = []
            label_dataframes[current_label].append(pd.DataFrame(sequence_rows))

        return label_dataframes[0], label_dataframes[1], label_dataframes[2], label_dataframes[3], label_dataframes[4], \
            label_dataframes[5]


class EvaluationUtility:

    @staticmethod
    def print_label_information(df: pd.DataFrame):
        sequence_counts = {}
        sequence_lengths = {}
        current_label = None
        current_sequence_count = 0

        for index, row in df.iterrows():
            label = row['label']
            if label != current_label:
                if current_label is not None:
                    if current_label in sequence_counts:
                        sequence_counts[current_label] += 1
                        sequence_lengths[current_label] += current_sequence_count
                    else:
                        sequence_counts[current_label] = 1
                        sequence_lengths[current_label] = current_sequence_count
                current_sequence_count = 0  # Reset the length counter for the new label
                current_label = label
            current_sequence_count += 1  # Increment the length counter

        print("Label statistics (raw data):")
        for label in sequence_counts:
            average_length = sequence_lengths[label] / sequence_counts[label]
            print(f"Label {label}: {sequence_counts[label]} sequences, Average Length: {average_length:.2f}")

    @staticmethod
    def print_model_cross_validation_scores(fitted_model, x_test, y_test, cv):
        """
        Returns the classification report for a model
        :param fitted_model: The fitted model
        :param x_test: The test features
        :param y_test: The test labels
        :param cv: The number of cross validation folds
        :return: The classification report
        """
        scores = cross_val_score(fitted_model, x_test, y_test, cv=cv)
        print("Cross Validation Scores")
        for i, score in enumerate(scores):
            print(f"Fold {i + 1}: {score}")

    @staticmethod
    def print_model_classification_report(fitted_model, x_test, y_test):
        """
        Returns the classification report for a model
        :param fitted_model: The fitted model
        :param x_test: The test features
        :param y_test: The test labels
        :return: The classification report
        """
        y_pred = fitted_model.predict(x_test)
        print(classification_report(y_test, y_pred))

    @staticmethod
    def plot_confusion_matrix(fitted_model, x_test, y_test, labels):
        """
        Plots a confusion matrix for a model
        :param fitted_model: The fitted model
        :param x_test: The test features
        :param y_test: The test labels
        :param labels: The label names
        """
        # Translate the labels.
        # (0.0 = Idle, 1.0 = Left Slip, 2.0 = Right Slip, 3.0 = Left Roll, 4.0 = Right Roll, 5.0 = Pull Back)
        yticklabels = ['Idle', 'L_Slip', 'R_Slip', 'L_Roll', 'R_Roll', 'Pull Back']
        labels = [yticklabels[int(label)] for label in labels]

        y_pred = fitted_model.predict(x_test)
        matrix = confusion_matrix(y_test, y_pred)
        sns.heatmap(matrix, annot=True, xticklabels=labels, yticklabels=labels)

    @staticmethod
    def evaluate_dtw(results_df, template_keys):
        """
        Plots the confusion matrix for the DTW classification results, and prints the classification report
        :param results_df: The results DataFrame (correct_class, predicted_class, dtw_distance, dtw_path)
        :param template_keys:  The keys of the templates used in the classification (gesture names)
        """
        # Generate the confusion matrix
        cm = confusion_matrix(results_df['correct_class'], results_df['predicted_class'], labels=list(template_keys))

        # Creating a DataFrame from the confusion matrix for better labeling
        cm_df = pd.DataFrame(cm, index=template_keys, columns=template_keys)

        # Plotting using seaborn
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', linewidths=0.5)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

        # Print the classification report
        print(classification_report(results_df['correct_class'], results_df['predicted_class'],
                                    target_names=template_keys, labels=list(template_keys)))

    @staticmethod
    def plot_feature_importances(fitted_model, x_test):
        """
        Plots the feature importances for a model
        :param fitted_model: The fitted model
        :param x_test: The test features
        """
        try:
            feature_importances = fitted_model.feature_importances_
            feature_importances, x_test.columns = zip(*sorted(zip(feature_importances, x_test.columns)))
            plt.figure(figsize=(10, 10))
            plt.barh(x_test.columns, feature_importances)
            plt.show()
        except:
            print("Model does not have feature importances")

    @staticmethod
    def plot_df_correlations(df: pd.DataFrame):
        """
        Plots the correlation matrix for a DataFrame
        :param df: The DataFrame
        """
        plt.figure(figsize=(20, 20))
        sns.heatmap(df.corr(), annot=True)
        plt.show()

    @staticmethod
    def plot_label_distribution(df: pd.DataFrame):
        """
        Plots the distribution of labels in a DataFrame with the count of each label annotated on the bars.
        :param df: The DataFrame containing the 'label' column.
        """
        plt.figure(figsize=(10, 6))
        ax = sns.histplot(df['label'], discrete=True)  # Ensure the plot treats the data as categorical
        plt.title('Label Distribution (Windowed Data)')

        # Annotate the counts above the bars
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                        textcoords='offset points')

        plt.show()

    @staticmethod
    def plot_sensor_data(title, dfs_to_plot):
        """
        Plots sensor data from one or multiple DataFrames into a single plot.

        :param title: The title of the plot
        :param dfs_to_plot: A single DataFrame or a list of DataFrames to plot,
                            each containing 'sensor_gyroY' and 'sensor_accX' columns
        """
        sensor_data = ['sensor_gyroY', 'sensor_accX']
        sensor_labels = ['gyroY', 'accX']
        ylabel = 'Raw Value'

        # Ensure dfs_to_plot is a list of DataFrames
        if isinstance(dfs_to_plot, pd.DataFrame):
            dfs_to_plot = [dfs_to_plot]

        # Define color palettes
        gyro_colors = ['#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4']
        acc_colors = ['#ff7f0e', '#ff7f0e', '#ff7f0e', '#ff7f0e']

        fig = go.Figure()

        for idx, df in enumerate(dfs_to_plot):
            for data, sensor_label, color in zip(sensor_data, sensor_labels, [gyro_colors[idx % len(gyro_colors)],
                                                                              acc_colors[idx % len(acc_colors)]]):
                label = f"(DF{idx + 1}) {sensor_label}"
                fig.add_trace(go.Scatter(y=df[data], mode='lines', name=label, line=dict(color=color)))

        fig.update_layout(
            title=title,
            xaxis_title='Sample',
            yaxis_title=ylabel,
            legend_title='Sensor Data',
            template='plotly_white',
        )
        fig.show()

    @staticmethod
    def plot_sensor_data_v2(title, gyro_y_data, acc_x_data):
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=gyro_y_data, mode='lines', name='Gyro Y', line=dict(color='blue')))
        fig.add_trace(go.Scatter(y=acc_x_data, mode='lines', name='Acc X', line=dict(color='orange')))
        fig.update_layout(
            title=title,
            xaxis_title='Sample',
            yaxis_title='Raw Value',
            legend_title='Sensor Data',
            template='plotly_white',
        )
        fig.show()

    @staticmethod
    def plot_sensor_data_v3(title, data_list):
        """
        :param title: The title of the plot
        :param data_list: A list of series to plot
        """
        fig = go.Figure()

        # Create a colormap
        colormap = cm.get_cmap('viridis', len(data_list))

        for idx, data in enumerate(data_list):
            # Generate a color from the colormap
            color = colormap(idx)
            # Convert the color from RGBA to an RGB format that Plotly accepts
            plotly_color = f'rgb({int(color[0] * 255)}, {int(color[1] * 255)}, {int(color[2] * 255)})'

            fig.add_trace(
                go.Scatter(y=data, mode='lines', name=f'Data {idx}', line=dict(color=plotly_color)))

        fig.update_layout(
            title=title,
            xaxis_title='Sample',
            yaxis_title='Raw Value',
            legend_title='Sensor Data',
            template='plotly_white',
        )
        fig.show()


class DynamicTimeWarpingUtility:

    @staticmethod
    def calculate_dtw_distance(sequence1: List[float], sequence2: List[float]):
        distance, path = fastdtw(sequence1, sequence2)
        return distance, path

    @staticmethod
    def classify_sequence(unknown_sequence: List[float], templates: dict):
        """
        Given a sequence, determines the most likely gesture using DTW
        :param unknown_sequence: The sequence to classify, as a list of floats
        :param templates: A dictionary with class names as keys and sequences as values (lists of floats / numpy arrays)
        :return:
        """
        # Prepare variables to store the shortest path and the predicted class
        lowest_distance = np.inf
        shortest_path = None
        predicted_class = None

        # Go over all the templates and calculate the DTW distance
        for template_class, template_data in templates.items():
            dtw_distance, dtw_path = DynamicTimeWarpingUtility.calculate_dtw_distance(sequence1=template_data,
                                                                                      sequence2=unknown_sequence)
            # Update the shortest path and the predicted class if the current distance is lower
            if dtw_distance < lowest_distance:
                lowest_distance = dtw_distance
                shortest_path = dtw_path
                predicted_class = template_class

        return predicted_class, lowest_distance, shortest_path

    @staticmethod
    def test_classification(dataframes_to_test: dict, templates: dict):
        """
        Tests the classification of sequences using DTW
        :param dataframes_to_test: A dictionary where the class names are keys, and the values are lists of dataframes
        :param templates: A dictionary with class names as keys and sequences as values (lists of floats / numpy arrays)
        :return: A DataFrame containing the results of the classification
        """
        results = []

        for correct_class, dataframes in dataframes_to_test.items():
            for dataframe in tqdm(dataframes, desc=f"Testing class {correct_class}", unit="dataframe"):
                # Extract the sensor data from the dataframe, and convert it to a list, in the correct order
                # Similarly to the template data (first all gyroY data, then all accX data, in one list)
                sequence = []
                sequence.extend(dataframe['sensor_gyroY'].values)
                sequence.extend(dataframe['sensor_accX'].values)

                predicted_class, dtw_distance, dtw_path = DynamicTimeWarpingUtility.classify_sequence(
                    unknown_sequence=sequence,
                    templates=templates)
                results.append({
                    'correct_class': correct_class,
                    'predicted_class': predicted_class,
                    'dtw_distance': dtw_distance,
                    'dtw_path': dtw_path,
                })

        return pd.DataFrame(results)

    @staticmethod
    def compute_dba_template(gesture_dataframes, sensor_type):
        """
        Compute the DBA template for a given list of gesture dataframes and a single sensor type
        :param gesture_dataframes: A list of dataframes, where a single dataframe represents a labeled sequence, so in others words, a gesture instance
        :param sensor_type: The sensor type for which the DBA template should be computed (will be extracted from the dataframe)
        :return:
        """
        series_array = []

        # Go over every occurrence (df) of the gesture, and extract the sensor data
        for df in gesture_dataframes:
            series = df[sensor_type].values  # Extract only the required sensor data from the dataframe
            series_array.append(series)

        # Compute DBA for the prepared sequences
        dba_template = dba.performDBA(series=series_array)

        return dba_template
