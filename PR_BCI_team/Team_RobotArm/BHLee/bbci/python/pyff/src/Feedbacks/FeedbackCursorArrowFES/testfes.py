import raweth
import sys

if __name__ == '__main__':
	print("FES test")
	eth = None
	try:
		eth = raweth.raweth('rawethdll.dll')
		eth.startup('{13904C50-488F-45BA-A4D8-0C721D384F46}', '00:E0:81:78:34:AC', 6300, 1, 4096)
		eth.add_participant('munduspc', '00:13:D3:AD:8E:0F', 6100, 1, 4096)
		eth.set_intensities(0.4,0.4,0.85,0.85)
		eth.set_stimulation_mode(0)
		eth.stimulate(6100, 1)	
		print("press RETURN to stop stimulation!")
		foo = sys.stdin.readline()
		print("stimulation stopped")
		eth.stimulate(6100, 666)	

	except:
		if eth is not None:
			self.eth.stimulate(6100, 666)	




