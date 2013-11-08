

import config 
def get_cuda_devices(compute_capability = config.compute_capability):
  try:
    import pycuda.autoinit
    import pycuda.driver  
  except:
    return []
  devices = [pycuda.driver.Device(i) for i in xrange(pycuda.driver.Device.count())]
  return [d for d in devices if d.compute_capability() >= compute_capability]

def device_id(cuda_device):
  import pycuda.driver 
  return cuda_device.get_attribute(pycuda.driver.device_attribute.PCI_DEVICE_ID)

def display_attached(cuda_device):
  import pycuda.driver 
  return cuda_device.get_attribute(pycuda.driver.device_attribute.KERNEL_EXEC_TIMEOUT) == 1  
  
def num_multiprocessors(cuda_device):
  import pycuda.driver
  return cuda_device.get_attribute(pycuda.driver.device_attribute.MULTIPROCESSOR_COUNT)
  

def best_cuda_device(compute_capability = config.compute_capability):
  devices = get_cuda_devices(compute_capability)
  if len(devices) == 0:
    return None
  
  best_device = devices[0]
  for d in devices[1:]:
    if display_attached(best_device) and not display_attached(d):
      best_device = d 
    elif num_multiprocessors(d) > num_multiprocessors(best_device):
      best_device = d
    elif d.total_memory() > best_device.total_memory():
      best_device = d
  return best_device    

def has_gpu():
  return best_cuda_device() is not None

    
  
  
  