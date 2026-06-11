# ⚡ Design and Implementation of Mechatronic Systems: ECU Test Bench

![Python Version](https://img.shields.io/badge/python-3.13.xx-blue.svg)
![Microcontroller](https://img.shields.io/badge/Microcontroller-ESP32-red.svg)
![Firmware](https://img.shields.io/badge/Firmware-MicroPython-yellow.svg)

This repository contains all the code, hardware schematics, and assets used for the class **Design and Implementation of Mechatronic Systems**. 

The core of this project is a **High-Current Load Controller (ECU Test Bench)** designed to safely simulate, control, and monitor high-power loads (12V/5A) using solid-state switching and a custom Python Graphical User Interface (GUI).

---# ⚡ Automated HIL Power & Grid Management Unit

> **Solid-State ECU Test Bench for Automotive Hardware-In-the-Loop (HIL) Validation**

This repository contains the firmware, host software, hardware schematics, and testing assets developed for the *Design and Implementation of Mechatronic Systems* engineering capstone. 

The core of this project is a **High-Current Power Management Unit (PMU)** designed to safely automate the power cycling, control, and real-time monitoring of automotive Electronic Control Units (ECUs). By replacing traditional mechanical relays and manual resets with a fully automated, VISA-compliant Python GUI and a solid-state switching matrix, this system provides a robust framework for continuous ECU stress-testing.

## 📋 System Architecture

The architecture relies on a strict separation of concerns across three layers, safely isolating the low-voltage logic control (3.3V) from the raw automotive power stage (12V) using galvanic isolation.

1. **Host Layer (PC / Python):** A robust GUI that manages serial connection watchdogs, visualizes live telemetry on an oscilloscope, and parses automation scripts (e.g., `LOOP 10`, `DELAY 100`) into single-line SCPI/VISA string commands (`*IDN?`, `ECU1_BAT ON`).
2. **Control Layer (ESP32 / MicroPython):** An asynchronous event-driven firmware (`asyncio`) that parses commands, manages GPIO logic, polls I2C sensors, and controls the SPI CAN bus simultaneously without blocking the main loop.
3. **Hardware Layer (Custom PCB):** A 12-channel power delivery matrix protecting the logic domain from automotive 12V inrush currents and inductive spikes.

## 🚀 Key Features & Core Components

* **Solid-State Power Matrix:** Independent 12V/5A control over Battery (BAT), Accessory (ACC), and Ignition (IGN) lines across 4 distinct ECU channels.
  * *Component:* P-Channel Power MOSFETs (IRF4905) driven by PC817 Optocouplers.
* **Real-Time I2C Telemetry:** Closed-loop monitoring of voltage, current, and power consumption per channel using custom bare-metal I2C drivers.
  * *Component:* INA226 digital power monitors with ultra-low 0.01Ω shunt resistors and hardware-configured I2C addresses.
* **CAN Bus Integration:** SPI-driven CAN controller integration for broadcast messaging, ECU node-health supervision, and live traffic monitoring via the GUI.
  * *Component:* MCP2515 CAN Controller + TJA1050 Transceiver.
* **Automated Sequencer:** Custom scripting engine allowing the execution of complex testing vectors (e.g. stress tests) directly from `.txt` files.

## 📂 Repository Structure

* `/Firmware`: MicroPython scripts for the ESP32 (main asynchronous loop, custom `INA226.py` and `mcp2515.py` register-level drivers, command parser).
* `/Software`: Python GUI application (Tkinter), live scope plotting, CAN monitor, and serial management.
* `/Hardware`: KiCad schematics, wiring diagrams, LTspice validation simulations (`.asc`, `.model`), and Bill of Materials (BOM).
* `/Tests`: Example sequence `.txt` files for automated stress testing (e.g., `COMPLETE Test.txt`).
* `/Docs`: Final Technical Report, architecture diagrams, and component datasheets.

## 🛠️ Setup and Installation

### 1. Hardware Assembly
Assemble the physical circuit following the schematics provided in the `/Hardware` folder.
> ⚠️ **Safety Warning:** This system handles cumulative currents capable of exceeding 15A during inrush events. Ensure the use of appropriate gauge wiring (AWG 16 minimum) for the 12V power bus, use proper thermal dissipation techniques, and include a master inline fuse & kill-switch before the MOSFET array.

### 2. Firmware (ESP32)
1. Flash the ESP32 microcontroller with the latest MicroPython firmware.
2. Upload all python scripts from the `/Firmware` folder to the board using tools like Thonny IDE or `mpremote`.
3. The board will automatically execute `main.py` on boot, initialize the SPI/I2C buses, and await serial commands at `115200` bauds.

### 3. Software (Python GUI)
The host control interface requires **Python 3.13.x**. Navigate to the software folder and install the required dependencies:

```bash
cd Software
pip install -r requirements.txt
python main.py
```
## 📋 System Architecture

The project is modularly designed, safely isolating the low-voltage logic control (3.3V) from the raw power stage (12V) using optocouplers. 

### Core Components:
* **Microcontroller:** ESP32 running MicroPython.
* **Power Switching:** P-Channel Power MOSFET (IRF4905).
* **Current/Voltage Monitoring:** INA226 (I2C) with an ultra-low shunt resistor.
* **Isolation:** PC817 Optocoupler.
* **Simulated Load:** 120 Ohms Resistors. (in total 12 Resistors ~1.2A drawn).

---

## 📂 Repository Structure

* `/firmware`: MicroPython scripts for the ESP32 (I2C sensor reading, PWM/Digital control).
* `/software`: Python GUI application for remote monitoring and control.
* `/hardware`: Circuit schematics, wiring diagrams, and LTspice simulations (`.asc`, `.model`).
* `/docs`: Final Report, technical documentation and component datasheets (INA226, IRF4905, etc.).

---

## 🛠️ Setup and Installation

### 1. Hardware Assembly
Assemble the physical circuit following the schematics provided in the `/hardware` folder. 
> ⚠️ **Safety Warning:** This system handles currents <= 5A. Ensure the use of appropriate gauge wiring (AWG 16 minimum) for the power stage and include a master kill-switch before the MOSFET array.

### 2. Firmware (ESP32)
Flash the ESP32 with the latest MicroPython firmware. Upload the scripts from the `/firmware` folder to the board using tools like Thonny IDE or `mpremote`.

### 3. Software (Python GUI)
The control interface requires **Python 3.13.xx**. Navigate to the software folder and install the required dependencies:

```bash
cd software
pip install -r requirements.txt
python main.py
