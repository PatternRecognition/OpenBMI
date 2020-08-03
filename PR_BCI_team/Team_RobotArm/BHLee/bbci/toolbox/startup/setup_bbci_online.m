%global EEG_VIDEO_DIR
%
%if isempty(EEG_VIDEO_DIR)
%  EEG_VIDEO_DIR = '/home/neuro/data/BCI/eegVideo/replays/';
%end

pathes = {'simulation','feedbacks','communication','communication/signalserver','log','online','subjects','training','utils','gui','setups','training/analyze','training/setups','training/finish','online/feedbacks','cmb_setups','gui/control_gui','replay','nogui'};

global BBCI_DIR BCI_DIR LOG_DIR TMP_DIR general_port_fields
if isempty(TMP_DIR),
  TMP_DIR= [DATA_DIR 'tmp/'];
end

BBCI_DIR = [BCI_DIR 'online/'];
for i = 1:length(pathes)
  path([BBCI_DIR pathes{i}], path);
end
