%send_cnt_like_bv emulates the BrainVision server
%
% description: 
%  The Function is a helper for testing online classifiers or something else
%  offline with some eeg-data you want.
%
% usage:
%     INIT: 
%        send_cnt_like_bv(cnt, 'init');
%     DATA:
%        pp= send_cnt_like_bv(cnt, pp);
%        pp= send_cnt_like_bv(cnt, pp, mrkPos,mrkDesc);
%
%     CLOSE:
%        send_cnt_like_bv('close');
%
% input:
%  INIT:
%    cnt         - the eeg data structure we use following fields:
%             cnt.fs - the sampling rate
%           cnt.clab - the channel names
%  DATA:
%    cnt          - the eeg data structure we use following fields:
%             cnt.x  - the eeg data
%    pp           - the position in the eeg data
%    mrkPos       - the positions of the markers
%    mrkDesc      - the description of the markers
%
% Compile:
%     make_send_cnt_like_bv
%
% AUTHOR
%    Max Sagebaum
%
%    2008/05/13 - Max Sagebaum
%                   - file created after refactoring of send_cnt_like_bv.c 
% (c) 2005 Fraunhofer FIRST