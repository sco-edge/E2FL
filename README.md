# E2FL

# USE FLOWER instead of FedML
https://github.com/adap/flower/tree/main/examples/embedded-devices

# Edge Device Setup (same as Flower)

## Setting up a Raspberry Pi
1. Installing Ubuntu server on your Raspberry Pi is easy with the Raspberry Pi Imager.
- Click on CHOOSE OS > Other general-purpose OS > Ubuntu > Ubuntu Server 22.04.03 LTS (64-bit). Other versions of Ubuntu Server would likely work but try to use a 64-bit one.
2. Connecting to your Raspberry Pi.
- After shh-ing into your Raspberry Pi for the first time, make sure your OS is up-to-date
'''
$ sudo apt update
$ sudo apt upgrade -y
$ sudo reboot
'''
3. Preparations for your flower experiments
'''
$ sudo apt install python3-pip -y
'''
## Setting up a Jetson Series


# Dependencies

## Flower
'''
$ git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/embedded-devices . && rm -rf flower && cd embedded-devices
$ pip3 install -r requirements_pytorch.txt
'''

## Linux Tool
- iw
- iperf3
- net-tools

'''
$ sudo apt install iw, iperf3 -y
'''

## Python Dependencies
- Monsoon Python library
- paramiko library
- cffi library
- yaml library
- pickle library


'''
$ pip install -r requirements_test_wrls-power.txt
'''
or
'''
$ pip install Monsoon, cffi, paramiko, yaml, pickle
'''

### Troubleshooting Monsoon Python Library 


# Test
