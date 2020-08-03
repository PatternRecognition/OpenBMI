function paradigm_name = get_paradigm_for_VP(VPDir)
% If VPDir is found in one of the session lists, this function returns a
% string that denotes the experiment during which the data was recorded.

% list of supported paradigms, with their session lists
paradigms = {
    % paradigm name, full session list name
    {'AMUSE', 'session_list_erp_AMUSE'}
    {'T9', 'session_list_erp_T9speller'}
    {'T9wharp', 'session_list_T9warp'}
    {'onlineVisualSpeller_HexoSpeller', 'session_list_onlineVisualSpeller_HexoSpeller'}
    {'RSVP', 'session_list_RSVP_online'}
    {'Vital_BCI', 'session_list_Vital_BCI'}
    {'MVEP', 'session_list_MVEP'}
};

% go through all paradigms
paradigm_name = [];
for p=1:length(paradigms)
    % get the session list for paradigm p
    vp_list = read_session_lists({paradigms{p}{2}});
    % check if VPDir is in the session list
    if any(ismember(vp_list, VPDir))
        paradigm_name = paradigms{p}{1};
    end
end