function [cnt, mrk, sbjData, opt] = load_online_data(VPDir, varargin)
%% 

if isstruct(VPDir)
    % for backwards compatibility ...
    opt = VPDir;
    VPDir = opt.VPDir;
end

% collect options
opt= propertylist2struct(varargin{:});


%% load online data, depending on experiment
exp_paradigm = get_paradigm_for_VP(VPDir);
switch exp_paradigm
    case 'T9'
        [cnt, mrk, sbjData] = load_online_data_T9(VPDir, opt);
    case 'T9wharp'
        [cnt, mrk, sbjData] = load_online_data_T9wharp(VPDir, opt);
    case 'AMUSE'
        [cnt, mrk, sbjData] = load_online_data_AMUSE(VPDir, opt);
    case 'onlineVisualSpeller_HexoSpeller'
        [cnt, mrk, sbjData] = load_online_data_HexoSpeller(VPDir, opt);
    case 'RSVP'
        [cnt, mrk, sbjData] = load_online_data_RSVP(VPDir, opt);
    case 'Vital_BCI'
        [cnt, mrk, sbjData] = load_online_data_Vital_BCI(VPDir, varargin{:});
    case 'MVEP',
        [cnt, mrk, sbjData] = load_online_data_MVEP(VPDir, varargin{:});        
        
    otherwise
        error('Unknown experiment paradigm!')
end


%% standard preprocessing
% i.e. filtering, artifact rejection
[cnt opt, mrk] = std_preprocessing(cnt, mrk, opt);

