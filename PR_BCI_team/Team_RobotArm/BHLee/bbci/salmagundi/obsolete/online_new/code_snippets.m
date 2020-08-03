% This was intended in the INIT case of bbci_apply_adaptation
% to allow seperate persistent var for different classifers
    if k>1,
      global TMP_DIR
      % TODO: make subfolder or use other directory
      addpath(TMP_DIR)
      fcn_name_orig= func2str(bbci.adaptation(1).fcn);
      fcn_name_copy= [fcn_name_orig '_no' int2str(k)];
      copyfile(which(fcn_name_orig), [TMP_DIR fcn_name_copy '.m']);
      bbci.adaptation(k).fcn= str2func(fcn_name_copy);
      BA= bbci.adaptation;
    end
