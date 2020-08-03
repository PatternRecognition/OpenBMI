pyff('startup','a','D:\development\mundus\runtime\pyff\src\Feedbacks', 'bvplugin',0 );

pyff('init','BGUI');
pyff('load_settings', 'D:\development\mundus\runtime\pyff\src\Feedbacks\BGUI\feedback_base');
pyff('load_settings', 'D:\development\mundus\runtime\pyff\src\Feedbacks\BGUI\feedback_online_MI_P300');
pyff('play')

bbci_acquire_bv('close')
bbci_online_multi('D:\data\bbciRaw\VPrp_12_05_18\', 'cfy_best_comb;cfy_idle', '0','0')