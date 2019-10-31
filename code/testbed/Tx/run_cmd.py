import os
import time

def run_cmd(skip_time):
	try:
		while True:
			cur_time = int(time.time())
			os.system('uhd_rx_cfile -f 882.5M --samp-rate=5000000 -m '+str(cur_time)+'_880M_5m.dat -N 200000000')
			time.sleep(skip_time*60)
			#os.system('mv '+str(cur_time)+'_880M_5m.dat /media/My\ Passport/ryperson_monitoring_round2/')
# /media/d6c0805f-bf60-4df1-a3f5-33541a776a47/monitoring
	except KeyboardInterrupt:
		pass

run_cmd(5)
