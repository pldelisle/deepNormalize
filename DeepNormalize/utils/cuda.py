import pycuda.driver as cuda
import pycuda.tools
import pycuda.autoinit

cuda.init()


class Cuda():
    def __init__(self):
        pass

    def num_devices(self):
        """Return number of devices connected."""
        return cuda.Device.count()

    def devices(self):
        """Get info on all devices connected."""
        num = cuda.Device.count()
        print("%d device(s) found:" % num)
        for i in range(num):
            print(cuda.Device(i).name(), "(Id: %d)" % i)

    def mem_info(self):
        """Get available and total memory of all devices."""
        available, total = cuda.mem_get_info()
        print("Available: %.2f GB\nTotal:     %.2f GB" % (available / 1e9, total / 1e9))

    def attributes(self, device_id=0):
        """Get attributes of device with device Id = device_id"""
        return cuda.Device(device_id).get_attributes()

    def __repr__(self):
        """Class representation as number of devices connected and about them."""
        num = cuda.Device.count()
        string = ""
        string += ("%d device(s) found:\n" % num)

        device_data = pycuda.tools.DeviceData()
        print("CUDA devices information:")
        for i in range(num):
            attributes = cuda.Device(i).get_attributes()

            print("%d) %s (Id: %d)" % ((i + 1), cuda.Device(i).name(), i))
            print("     Memory: %.2f GB" % (cuda.Device(i).total_memory() / 1e9))
            print("     Attributes for device ID %d" % i)
            for (key, value) in attributes.items():
                print("         %s:%s" % (str(key).lower(), str(value)))
        return string
