bvr_sendcommand('checkimpedances');
fprintf('Prepare cap. Press <RETURN> when finished.\n');
pause

VP_SCREEN = [-1280 0 1280 1024];
VP_CODE = 'VPzq';
CLSTAG = 'LR';

%-newblock
setup_intelligent_entscheiden_imag_arrow;
fprintf('Press <RETURN> when ready to start ''imagined movements''.\n');
pause
stim_visualCues(stim, opt);
fprintf('Press <RETURN> when ready to start the secon run.\n');
pause

%-newblock
setup_intelligent_entscheiden_imag_arrow;
fprintf('Press <RETURN> when ready to start ''imagined movements''.\n');
pause
stim_visualCues(stim, opt);
fprintf('Press <RETURN> when ready to start the recording.\n');
pause

cmd= sprintf('CLSTAG= ''%s''; VP_CODE= ''%s''; ', CLSTAG, VP_CODE);

fprintf('You still have to do:\n');
fprintf('bbci_bet_prepare\n');
fprintf('bbci_bet_analyze\n');
fprintf('bbci_bet_finish\n');
fprintf('bbci_bet_apply\n');

return

%% for brain pong feedback with pyff
system('cmd /C "d: & cd \svn\pyff\src & python FeedbackController.py -l debug -p FeedbackControllerPlugins" &')
%%general_port_fields.feedback_receiver= 'pyff';
%%bbci_bet_apply(bbci.save_name, 'bbci.feedback', '1d', 'bbci.fb_port', 12345);