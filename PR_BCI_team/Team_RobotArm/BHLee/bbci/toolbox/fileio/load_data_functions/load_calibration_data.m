function [cnt, mrk, sbjData, opt] = load_calibration_data(VPDir, varargin)
%% 

% % collect options
% opt= propertylist2struct(varargin{:});


%% load calibration data, depending on experiment
exp_paradigm = get_paradigm_for_VP(VPDir);
switch exp_paradigm
    case 'T9'
        [cnt, mrk, sbjData] = load_calibration_data_T9(VPDir, varargin{:});
    case 'T9wharp'
        [cnt, mrk, sbjData] = load_calibration_data_T9wharp(VPDir, varargin{:});
    case 'AMUSE'
        [cnt, mrk, sbjData] = load_calibration_data_AMUSE(VPDir, varargin{:});
    case 'onlineVisualSpeller_HexoSpeller'
        [cnt, mrk, sbjData] = load_calibration_data_HexoSpeller(VPDir, varargin{:});
    case 'RSVP'
        [cnt, mrk, sbjData] = load_calibration_data_RSVP(VPDir, varargin{:});
    case 'Vital_BCI'
        [cnt, mrk, sbjData] = load_calibration_data_Vital_BCI(VPDir, varargin{:});
    case 'MVEP',
        [cnt, mrk, sbjData] = load_calibration_data_MVEP(VPDir, varargin{:});
    
    otherwise
        error('Unknown experiment paradigm!')
end


%% standard preprocessing
% i.e. filtering, artifact rejection
[cnt opt, mrk] = std_preprocessing(cnt, mrk, varargin{:});

