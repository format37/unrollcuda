import unrollcuda as uc
import psutil


def main():
    # Define the unrollcuda instance
    ker = uc.kernel()

    # Cuda device
    print('GPU:', ker.dev.name())

    # Cuda memory
    gpu_memory = ker.drv.mem_get_info()
    gpu_memory_available = gpu_memory[0]/1024**3
    # Format to x.xx
    gpu_memory_available = "{:.2f}".format(gpu_memory_available)
    print('GPU memory available: ', gpu_memory_available)

    # RAM memory
    ram_memory = psutil.virtual_memory().available/1024**3
    # Format to x.xx
    ram_memory = "{:.2f}".format(ram_memory)
    print('RAM memory available: ',ram_memory,'Gb')

if __name__ == '__main__':
    main()
