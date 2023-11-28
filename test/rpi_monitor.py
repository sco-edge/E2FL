import serial
import time


def execute_shell_script(serial_port, script_path):
    ser = serial.Serial(serial_port, baudrate=9600, timeout=1)

    try:
        command = f'bash {script_path}\n'
        ser.write(command.encode('utf-8'))
        time.sleep(1)
    
        response = ser.read(ser.in_waiting).decode('utf-8')
        print("Shell Script Execution Result")
        print(response)
    except Exception as e:
        print(f"An error occurred: {e}")    

    finally:
        ser.close()
        

if __name__ == "__main__":
    # Configuration
    rpi_serial_port = "/dev/ttyUSB0"
    shell_path = "/home/E2FL/test/"
    
    execute_shell_script(rpi_serial_port, shell_path)


