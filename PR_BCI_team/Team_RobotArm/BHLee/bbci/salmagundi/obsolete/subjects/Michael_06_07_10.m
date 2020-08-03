calib= {'lettVPcv','lettVPcv2','moveVPcv','moveVPcv2'};
bbci.train_file= strcat('VPcv_06_07_10/imag_', calib);
bbci.classDef = {1,2,3;'left','right','foot'};
bbci.player = 2;
bbci.setup = 'csp';
bbci.save_name = 'Michael_06_07_10/imag_Michael';
bbci.feedback = '1d';
bbci.classes = {'left','right'};
