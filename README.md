# ⚡ Design and Implementation of Mechatronic Systems: ECU Test Bench

![Python Version](https://img.shields.io/badge/python-3.13.xx-blue.svg)
![Microcontroller](https://img.shields.io/badge/Microcontroller-ESP32-red.svg)
![Firmware](https://img.shields.io/badge/Firmware-MicroPython-yellow.svg)

This repository contains all the code, hardware schematics, and assets used for the class **Design and Implementation of Mechatronic Systems**. 

The core of this project is a **High-Current Load Controller (ECU Test Bench)** designed to safely simulate, control, and monitor high-power loads (12V/5A) using solid-state switching and a custom Python Graphical User Interface (GUI).

---

## 📋 System Architecture

The project is modularly designed, safely isolating the low-voltage logic control (3.3V) from the raw power stage (12V) using optocouplers. 

### Core Components:
* **Microcontroller:** ESP32 running MicroPython.
* **Power Switching:** P-Channel Power MOSFET (IRF4905).
* **Current/Voltage Monitoring:** INA226 (I2C) with an ultra-low shunt resistor.
* **Isolation:** PC817 Optocoupler.
* **Simulated Load:** 55W H4 Halogen Bulb (~5A draw).

---

## 📂 Repository Structure

* 📁 `/firmware`: MicroPython scripts for the ESP32 (I2C sensor reading, PWM/Digital control).
* 📁 `/software`: Python GUI application for remote monitoring and control.
* 📁 `/hardware`: Circuit schematics, wiring diagrams, and LTspice simulations (`.asc`, `.model`).
* 📁 `/docs`: Final Report, technical documentation and component datasheets (INA226, IRF4905, etc.).

---

## 🛠️ Setup and Installation

### 1. Hardware Assembly
Assemble the physical circuit following the schematics provided in the `/hardware` folder. 
> ⚠️ **Safety Warning:** This system handles currents >5A. Ensure the use of appropriate gauge wiring (AWG 16 minimum) for the power stage and include a master kill-switch before the MOSFET array.

### 2. Firmware (ESP32)
Flash the ESP32 with the latest MicroPython firmware. Upload the scripts from the `/firmware` folder to the board using tools like Thonny IDE or `mpremote`.

### 3. Software (Python GUI)
The control interface requires **Python 3.13.xx**. Navigate to the software folder and install the required dependencies:

```bash
cd software
pip install -r requirements.txt
python main.py