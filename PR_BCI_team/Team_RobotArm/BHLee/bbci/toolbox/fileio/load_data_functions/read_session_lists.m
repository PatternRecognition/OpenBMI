function vp_list = read_session_lists(session_lists, session_list_folder)

global BCI_DIR
    
if nargin < 2
    session_list_folder = fullfile(BCI_DIR,'toolbox','fileio',...
        'load_data_functions','session_lists');
end

% load the vp session lists and extract the vp folder names
vp_list = {};
for file_idx=1:length(session_lists)
    vp_list_path = fullfile(session_list_folder, session_lists{file_idx});
    tmp_vp_list = textread(vp_list_path, '%q');
    vp_list = [vp_list; tmp_vp_list];
end
