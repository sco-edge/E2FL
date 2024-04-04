from power import Monitor

node_A_name = 'rpi3B+'
node_A_vout = 5.0
node_A_mode = "PyMonsoon"
node_A_triggerBool = True
node_A_numSamples = 5000
node_A_thld_high = 100
node_A_thld_low = 10

rpi3B = Monitor.PowerMon(   node = node_A_name,
                            vout = node_A_vout,
                            mode = node_A_mode )

rpi3B.setTrigger(   bool = node_A_triggerBool,
                    numSamples = node_A_numSamples,
                    thld_high = node_A_thld_high,
                    thld_low = node_A_thld_low )
