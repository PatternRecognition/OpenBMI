function vp = get_vp_from_session_list(session_list)
%% GET_VP_FROM_SESSION_LIST
% vplist = get_vp_from_session_list(session_list)
%
% INPUT: 
%    session_list: cell-array with sessions that you get with e.g. 
%    session_list = get_session_list('season9')
%
% Johannes 02/2011
vp = {};
n=length(session_list);
for ii = 1:n
    dum = strfind(session_list{ii} , '_');
    vp{ii} = session_list{ii}(1:dum(1)-1);
end
vp = vp';