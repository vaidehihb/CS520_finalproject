import math
import numpy as np

if __name__ == '__main__':
	a = [36.3,29.7,28.5,27.5,27.6,25.4,25.1,24.5,24.4,24.3]
	print(np.std(a, dtype=np.float64))
	
	a = [51.34,38,28,22.67,18,17.34,16,14.67,14,11.34]
	print(np.std(a, dtype=np.float64))
	
	a = [88.4,62.8,59.9,65.9,53.2,53.3,61,56.5,48.3,45.4]
	print(np.std(a, dtype=np.float64))
	
	a = [48.67,48.67,51.34,51.34,48.67,48.67,51.34,16.67,48.67,14.67]
	print(np.std(a, dtype=np.float64))
	
	a = [40,25,26.67,25,16,16.67,18.67,17.5,17.78,17]
	print(np.std(a, dtype=np.float64))
	
	a = [42.67,32.67,48.67,48.67,46,44.67,44.67,45.34,43.34,42.67]
	print(np.std(a, dtype=np.float64))