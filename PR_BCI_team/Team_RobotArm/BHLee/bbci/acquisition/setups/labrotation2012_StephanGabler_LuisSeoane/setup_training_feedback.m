data_path = 'c:\Dokumente und Einstellungen\ml\Eigene Dateien\stephan_luis\data\190412_145725\';
pyff('init', 'TrainingFeedback');
pause(2);
pyff('setint', 'n_groups', 4);
pyff('setint', 'group_size', 6);
pyff('set', 'prune', 0.15);
pyff('setint', 'n_bursts', 90);
pyff('setint', 'training_interval', 130);
pyff('set', 'data_path', data_path);
pyff('set', 'debug_path', [TODAY_DIR 'test.log'])
pyff('setint', 'geometry', VP_SCREEN);
t = clock;
t_stamp = [num2str(t(4)) '_' num2str(t(5)) '.log'];
pyff('set', 'debug_path', [TODAY_DIR 'cali_' t_stamp]);