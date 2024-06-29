import asyncio
import struct
import sys
from io import StringIO

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

sys.coinit_flags = 0  # Needed to avoid Windows bug  https://bleak.readthedocs.io/en/latest/troubleshooting.html
from BoxingRecognition.util import Window
from BoxingRecognition.util import SlidingWindow, DynamicTimeWarpingUtility

from bleak import BleakClient, BleakScanner, BleakError
from typing import List, Dict, Any, Optional

from colorama import init as colorama_init
from colorama import Fore
from colorama import Style

colorama_init()


class Component:
    def __init__(self, type: int, group_name: str, component_name: str, unit_name: str):
        self.type = type
        self.group_name = group_name
        self.component_name = component_name
        self.unit_name = unit_name

    def __str__(self):
        return f'Component(type: {self.type}, group_name: {self.group_name}, component_name: {self.component_name}, unit_name: {self.unit_name})'


class SensorScheme:
    def __init__(self, sensor_id: int, sensor_name: str, component_count: int):
        self.sensor_id = sensor_id
        self.sensor_name = sensor_name
        self.component_count = component_count
        self.components: List[Component] = []

    def __str__(self):
        return f'SensorScheme(sensor_id: {self.sensor_id}, sensor_name: {self.sensor_name}, components: {[str(component) for component in self.components]})'


class OpenEarableSensorConfig:
    def __init__(self, sensor_id: int, sampling_rate: float, latency: int):
        self.sensor_id = sensor_id
        self.sampling_rate = sampling_rate
        self.latency = latency

    @property
    def byte_list(self) -> List[int]:
        data = struct.pack('<BfI', self.sensor_id, self.sampling_rate, self.latency)
        return list(data)

    def __str__(self):
        return f'OpenEarableSensorConfig(sensor_id: {self.sensor_id}, sampling_rate: {self.sampling_rate}, latency: {self.latency})'


class BleManager:
    sensorServiceUuid = "34c2e3bb-34aa-11eb-adc1-0242ac120002"
    sensorConfigurationCharacteristicUuid = "34c2e3bd-34aa-11eb-adc1-0242ac120002"
    sensorDataCharacteristicUuid = "34c2e3bc-34aa-11eb-adc1-0242ac120002"

    def __init__(self):
        self.mtu = 60
        self.client: Optional[BleakClient] = None
        self.connected_device: Optional[str] = None
        self.device_identifier: Optional[str] = None
        self.device_firmware_version: Optional[str] = None
        self.device_hardware_version: Optional[str] = None

    async def start_scan(self):
        print("Scanning for OpenEarable devices...")
        devices = await BleakScanner.discover()
        return devices

    async def connect_to_device(self, address: str):
        try:
            self.client = BleakClient(address)
            await self.client.connect()
            self.connected_device = address
            print(f"Connected to {address}")
            await self._configure_device()
        except BleakError as e:
            print(f"Failed to connect: {str(e)}")
            self.client = None

    async def _configure_device(self):
        if self.client and self.client.is_connected:
            try:
                await self.read_device_info()
            except Exception as e:
                print(f"Failed to configure device: {str(e)}")

    async def disconnect(self):
        if self.client:
            await self.client.disconnect()
            self.client = None
            self.connected_device = None
            print("Disconnected")

    async def read_characteristic(self, characteristic_uuid: str) -> List[int]:
        if self.client and self.client.is_connected:
            data = await self.client.read_gatt_char(characteristic_uuid)
            return list(data)
        else:
            raise Exception("No device is connected")

    async def write_characteristic(self, characteristic_uuid: str, value: List[int]):
        if self.client and self.client.is_connected:
            await self.client.write_gatt_char(characteristic_uuid, bytearray(value), response=True)
            print("Write successful")
        else:
            raise Exception("No device is connected")

    async def subscribe_to_characteristic(self, characteristic_uuid: str, handler):
        if self.client and self.client.is_connected:
            await self.client.start_notify(characteristic_uuid, handler)
        else:
            raise Exception("No device is connected")

    async def read_device_info(self):
        if not self.client or not self.client.is_connected:
            raise Exception("No device is connected")

        try:
            device_info_uuids = {
                'identifier': "45622511-6468-465a-b141-0b9b0f96b468",
                'firmware_version': "45622512-6468-465a-b141-0b9b0f96b468",
                'hardware_version': "45622513-6468-465a-b141-0b9b0f96b468"
            }
            self.device_identifier = bytes(await self.read_characteristic(device_info_uuids['identifier'])).decode(
                'utf-8')
            self.device_firmware_version = bytes(
                await self.read_characteristic(device_info_uuids['firmware_version'])).decode('utf-8')
            self.device_hardware_version = bytes(
                await self.read_characteristic(device_info_uuids['hardware_version'])).decode('utf-8')

            print("Device Identifier:", self.device_identifier)
            print("Device Firmware Version:", self.device_firmware_version)
            print("Device Hardware Version:", self.device_hardware_version)
        except Exception as e:
            print(f"Failed to read device info: {str(e)}")

    async def read(self, service_id: str, characteristic_id: str) -> List[int]:
        if not self.client or not self.client.is_connected:
            raise Exception("No device is connected")
        return await self.read_characteristic(characteristic_id)

    @property
    def connected(self) -> bool:
        return self.client is not None and self.client.is_connected


class SensorManager:
    sensorServiceUuid = "34c2e3bb-34aa-11eb-adc1-0242ac120002"
    sensorConfigurationCharacteristicUuid = "34c2e3bd-34aa-11eb-adc1-0242ac120002"
    sensorDataCharacteristicUuid = "34c2e3bc-34aa-11eb-adc1-0242ac120002"
    batteryServiceUuid = "0000180f-0000-1000-8000-00805f9b34fb"
    batteryLevelCharacteristicUuid = "00002a19-0000-1000-8000-00805f9b34fb"
    batteryStateCharacteristicUuid = "00002a1a-0000-1000-8000-00805f9b34fb"
    buttonServiceUuid = "29c10bdc-4773-11ee-be56-0242ac120002"
    buttonStateCharacteristicUuid = "29c10f38-4773-11ee-be56-0242ac120002"
    parseInfoServiceUuid = "caa25cb7-7e1b-44f2-adc9-e8c06c9ced43"
    schemeCharacteristicUuid = "caa25cb8-7e1b-44f2-adc9-e8c06c9ced43"

    def __init__(self, ble_manager: BleManager):
        self.imuID = 0
        self._ble_manager = ble_manager
        self._sensor_schemes: Optional[List[SensorScheme]] = None

    async def write_sensor_config(self, sensor_config: OpenEarableSensorConfig):
        if not self._ble_manager.connected:
            raise Exception("Can't write sensor config. Earable not connected")
        await self._ble_manager.write_characteristic(
            self.sensorConfigurationCharacteristicUuid,
            sensor_config.byte_list
        )
        if self._sensor_schemes is None:
            await self._read_sensor_scheme()

    async def subscribe_to_sensor_data(self, sensor_id: int) -> asyncio.Queue:
        if not self._ble_manager.connected:
            raise Exception("Can't subscribe to sensor data. Earable not connected")
        queue = asyncio.Queue()
        last_timestamp = 0

        async def notification_handler(characteristic, data):
            nonlocal last_timestamp
            if data and data[0] == sensor_id:
                parsed_data = await self._parse_data(data)
                if sensor_id == self.imuID:
                    timestamp = parsed_data["timestamp"]
                    last_timestamp = timestamp
                await queue.put(parsed_data)

        await self._ble_manager.subscribe_to_characteristic(
            self.sensorDataCharacteristicUuid,
            notification_handler
        )
        return queue

    async def _parse_data(self, data: List[int]) -> Dict[str, Any]:
        byte_data = struct.pack('<' + 'B' * len(data), *data)
        byte_index = 0
        sensor_id, = struct.unpack_from('<B', byte_data, byte_index)
        byte_index += 2
        timestamp, = struct.unpack_from('<I', byte_data, byte_index)
        byte_index += 4
        parsed_data = {"sensorId": sensor_id, "timestamp": timestamp}

        if self._sensor_schemes is None:
            await self._read_sensor_scheme()

        found_scheme = next(scheme for scheme in self._sensor_schemes if scheme.sensor_id == sensor_id)
        parsed_data["sensorName"] = found_scheme.sensor_name

        for component in found_scheme.components:
            if component.group_name not in parsed_data:
                parsed_data[component.group_name] = {}
                parsed_data[component.group_name]["units"] = {}

            parsed_value = None
            if component.type == 0:
                parsed_value, = struct.unpack_from('<b', byte_data, byte_index)
                byte_index += 1
            elif component.type == 1:
                parsed_value, = struct.unpack_from('<B', byte_data, byte_index)
                byte_index += 1
            elif component.type == 2:
                parsed_value, = struct.unpack_from('<h', byte_data, byte_index)
                byte_index += 2
            elif component.type == 3:
                parsed_value, = struct.unpack_from('<H', byte_data, byte_index)
                byte_index += 2
            elif component.type == 4:
                parsed_value, = struct.unpack_from('<i', byte_data, byte_index)
                byte_index += 4
            elif component.type == 5:
                parsed_value, = struct.unpack_from('<I', byte_data, byte_index)
                byte_index += 4
            elif component.type == 6:
                parsed_value, = struct.unpack_from('<f', byte_data, byte_index)
                byte_index += 4
            elif component.type == 7:
                parsed_value, = struct.unpack_from('<d', byte_data, byte_index)
                byte_index += 8

            parsed_data[component.group_name][component.component_name] = parsed_value
            parsed_data[component.group_name]["units"][component.component_name] = component.unit_name

        return parsed_data

    async def get_battery_level_stream(self) -> asyncio.Queue:
        return await self._subscribe_to_stream(self.batteryServiceUuid, self.batteryLevelCharacteristicUuid)

    async def get_button_state_stream(self) -> asyncio.Queue:
        return await self._subscribe_to_stream(self.buttonServiceUuid, self.buttonStateCharacteristicUuid)

    async def _subscribe_to_stream(self, service_id: str, characteristic_id: str) -> asyncio.Queue:
        queue = asyncio.Queue()

        async def notification_handler(characteristic, data):
            await queue.put(data)

        await self._ble_manager.subscribe_to_characteristic(
            characteristic_id,
            notification_handler
        )
        return queue

    async def _read_sensor_scheme(self):
        byte_stream = await self._ble_manager.read(
            self.parseInfoServiceUuid,
            self.schemeCharacteristicUuid
        )
        current_index = 0
        num_sensors = byte_stream[current_index]
        current_index += 1
        sensor_schemes = []

        for _ in range(num_sensors):
            sensor_id = byte_stream[current_index]
            current_index += 1
            name_length = byte_stream[current_index]
            current_index += 1
            sensor_name = bytes(byte_stream[current_index:current_index + name_length]).decode('utf-8')
            current_index += name_length
            component_count = byte_stream[current_index]
            current_index += 1

            sensor_scheme = SensorScheme(sensor_id, sensor_name, component_count)
            for _ in range(component_count):
                component_type = byte_stream[current_index]
                current_index += 1
                group_name_length = byte_stream[current_index]
                current_index += 1
                group_name = bytes(byte_stream[current_index:current_index + group_name_length]).decode('utf-8')
                current_index += group_name_length
                component_name_length = byte_stream[current_index]
                current_index += 1
                component_name = bytes(byte_stream[current_index:current_index + component_name_length]).decode('utf-8')
                current_index += component_name_length
                unit_name_length = byte_stream[current_index]
                current_index += 1
                unit_name = bytes(byte_stream[current_index:current_index + unit_name_length]).decode('utf-8')
                current_index += unit_name_length

                component = Component(component_type, group_name, component_name, unit_name)
                sensor_scheme.components.append(component)

            sensor_schemes.append(sensor_scheme)

        self._sensor_schemes = sensor_schemes


async def select_device_and_connect(ble_manager):
    print("Scanning for BLE devices...")
    devices = await BleakScanner.discover()

    if not devices:
        print("No devices found. Please make sure your BLE device is in range and try again.")
        return

    for idx, device in enumerate(devices):
        # Only print the instance if the name contains 'OpenEarable'
        if device.name is not None and 'OpenEarable' in device.name:
            print(f"{idx + 1}: {device.name} (Address: {device.address})")

    while True:
        choice = input("Enter the number of the device you want to connect to: ")
        if choice.isdigit() and 1 <= int(choice) <= len(devices):
            selected_device = devices[int(choice) - 1]
            break
        else:
            print("Invalid input, please enter a number from the list.")

    print(f"Connecting to {selected_device.name}...")
    await ble_manager.connect_to_device(selected_device.address)
    print(f"Connected to {selected_device.name} at {selected_device.address}")

    # Optionally perform further operations
    await ble_manager.read_device_info()


async def main():
    # Create a BLE manager and connect to a device
    ble_manager = BleManager()
    await select_device_and_connect(ble_manager)

    # Create a sensor manager and write sensor configuration
    sensor_manager = SensorManager(ble_manager)
    sensor_config = OpenEarableSensorConfig(sensor_id=0, sampling_rate=50, latency=0)
    await sensor_manager.write_sensor_config(sensor_config)

    # Subscribe to sensor data stream
    sensor_data_queue = await sensor_manager.subscribe_to_sensor_data(sensor_id=0)

    # Queue for processing data
    process_queue = asyncio.Queue()

    # Start the data processing task
    processor_task = asyncio.create_task(process_data_v2(process_queue))

    # Count the number of sensor data received
    counter = 0
    while True:
        data = await sensor_data_queue.get()  # Blocking get from the sensor data queue
        counter += 1  # Increment the counter
        await process_queue.put(data)  # Non-blocking put to the processing queue


model, feature_names = joblib.load("../BoxingRecognition/models/decision_tree_model_and_features.pkl")


async def process_data(queue):
    # Prepare the windows
    window_size = 50
    overlap_size = 25
    sliding_accelerometer_window = SlidingWindow(sensor_type='accelerometer', window_size=window_size,
                                                 overlap_size=overlap_size)
    sliding_gyroscope_window = SlidingWindow(sensor_type='gyroscope', window_size=window_size,
                                             overlap_size=overlap_size)

    # Receive data from the queue
    collecting_data = True
    while True:
        data = await queue.get()

        # Check if the data hits the gyro_y threshold of abs(3)
        if abs(data['GYRO']['Y']) > 3 or collecting_data:
            # Hit, that means that we have to fill an entire window, until we hit the threshold again
            collecting_data = True
            acc_window = sliding_accelerometer_window.add_data(data['ACC']['X'], data['ACC']['Y'], data['ACC']['Z'],
                                                               None)
            gyro_window = sliding_gyroscope_window.add_data(data['GYRO']['X'], data['GYRO']['Y'], data['GYRO']['Z'],
                                                            None)

            # Check if the windows are full
            if acc_window is not None and gyro_window is not None:
                # Schedule a prediction as a non-blocking task
                task = asyncio.create_task(make_prediction(acc_window, gyro_window))

                # Reset the collecting_data flag
                # collecting_data = False


async def process_data_v2(queue):
    # Anomaly Detection: Start & End thresholds for the gyroscope y-axis
    gyroscope_y_threshold = 1.5
    anomaly_end_threshold_low = -1
    anomaly_end_threshold_high = 1

    # Anomaly Detection: Parameters
    start_threshold_count = 1  # Number of consecutive samples above the threshold to start collecting data
    minimum_anomaly_length = 20  # Minimum number of samples to consider an anomaly
    counter_threshold = 5  # Number of consecutive samples within the end threshold to stop collecting data

    # Variables to keep track of the current state
    acc_sequence = []
    gyro_sequence = []
    collecting_data = False
    in_threshold_counter = 0
    above_threshold_counter = 0

    while True:
        # Get the data from the queue
        data = await queue.get()

        # Anomaly detection
        if abs(data['GYRO']['Y']) > gyroscope_y_threshold:
            above_threshold_counter += 1
        else:
            above_threshold_counter = 0  # Reset if the value falls below threshold at any point

        # If anomaly is detected, start collecting data
        if above_threshold_counter >= start_threshold_count:
            if not collecting_data:
                collecting_data = True
                print("\nAnomaly found: collecting data.")
                # Start of anomaly detected: reset sequences and counters
                acc_sequence = []
                gyro_sequence = []
                in_threshold_counter = 0  # Reset end threshold counter

        # Mid-anomaly data collection
        if collecting_data:
            gyro_sequence.append(-data['GYRO']['Y'])  # Data is inverted, revert it back
            acc_sequence.append(data['ACC']['X'])

            # Check if the data falls within the end thresholds
            if anomaly_end_threshold_low <= data['GYRO']['Y'] <= anomaly_end_threshold_high:
                in_threshold_counter += 1
            else:
                in_threshold_counter = 0  # Reset counter if data goes out of threshold range

            # Stop collecting data if the condition of 5 samples within threshold is met
            if in_threshold_counter >= counter_threshold:
                collecting_data = False
                if len(gyro_sequence) >= minimum_anomaly_length:
                    # Schedule a prediction as a non-blocking task
                    print(f"gyroscope_y_sequence = {gyro_sequence} \n"
                          f"accelerometer_x_sequence = {acc_sequence}")
                    # Append the two sequences together
                    unknown_sequence = gyro_sequence + acc_sequence
                    task = asyncio.create_task(make_prediction_v2(unknown_sequence))
                else:
                    print(f"Anomaly ended. Data collected was too short ({len(gyro_sequence)} samples).")
                # Reset sequences and counters for next detection
                acc_sequence = []
                gyro_sequence = []


# Import the templates
left_slip_template = np.loadtxt('../Data/dtw/left_slip_concatenated_template.csv', delimiter=',')
right_slip_template = np.loadtxt('../Data/dtw/right_slip_concatenated_template.csv', delimiter=',')
left_roll_template = np.loadtxt('../Data/dtw/left_roll_concatenated_template.csv', delimiter=',')
right_roll_template = np.loadtxt('../Data/dtw/right_roll_concatenated_template.csv', delimiter=',')
pull_back_template = np.loadtxt('../Data/dtw/pull_back_concatenated_template.csv', delimiter=',')
templates = {
    'left_slip': left_slip_template,
    'right_slip': right_slip_template,
    'left_roll': left_roll_template,
    'right_roll': right_roll_template,
    'pull_back': pull_back_template
}


async def make_prediction_v2(unknown_sequence: List[float]):
    plot_ascii(unknown_sequence)
    # Make the prediction
    prediction, distance, path = DynamicTimeWarpingUtility.classify_sequence(unknown_sequence=unknown_sequence,
                                                                             templates=templates)
    print(f"Prediction: {Fore.GREEN}{prediction}{Style.RESET_ALL} (Distance: {distance})")


async def make_prediction(accelerometer_window: Window, gyroscope_window: Window):
    # Get the feature row to predict (Without the label and window_id columns)
    acc_features = accelerometer_window.extract_features()
    gyro_features = gyroscope_window.extract_features()

    # Create a row for the DataFrame
    row = {**acc_features.iloc[0].to_dict(), **gyro_features.iloc[0].to_dict()}

    # Create a pandas dataframe, with the keys as the column names, and the values as the row values
    df = pd.DataFrame([row])

    # Make the prediction
    prediction = model.predict(df)[0]
    pred_num = prediction
    # Translate the prediction (0=idle, 1=left slip, 2=right slip, 3=left roll, 4=right roll, 5 = pull back)
    if prediction == 0:
        prediction = "Idle"
    elif prediction == 1:
        prediction = "Left Slip"
    elif prediction == 2:
        prediction = "Right Slip"
    elif prediction == 3:
        prediction = "Left Roll"
    elif prediction == 4:
        prediction = "Right Roll"
    elif prediction == 5:
        prediction = "Pull Back"
    else:
        prediction = "Unknown (Error)"

    print(f"\nPrediction: {Fore.GREEN}{prediction}{Style.RESET_ALL} ({pred_num}) \n")


def plot_ascii(series):
    # Capture the plot's output as a string (ASCII art)
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    plt.figure(figsize=(10, 4))  # Set figure size
    plt.plot(series)  # Plot the series
    plt.title("ASCII Plot")
    plt.grid(True)  # Enable grid for better readability
    plt.show()  # This will print the plot as text

    # Retrieve the ASCII plot from the buffer
    ascii_plot = sys.stdout.getvalue()

    # Reset stdout
    sys.stdout = old_stdout

    # Print the ASCII plot
    print(ascii_plot)


if __name__ == '__main__':
    asyncio.run(main())
