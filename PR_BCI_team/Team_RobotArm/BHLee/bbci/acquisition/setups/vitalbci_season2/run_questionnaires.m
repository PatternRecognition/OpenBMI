session_name= 'vitalbci_season2';
acqFolder = [BCI_DIR 'acquisition/setups/' session_name '/'];
questionnaires= strcat(acqFolder, ...
                       {'02_BIS-15.htm', ...
                        '03_KUT_Fragebogen_5Skalenstufen.htm', ...
                        '04_ADS-L.htm', ...
                        '05_Fragebogen_zur_aktuellen_Stimmung.htm', ...
                        '01_Daten_zu_Ihrer_Person.htm'});

pyff('startup','a',[BCI_DIR 'python\pyff\src\Feedbacks'], 'bvplugin', 0);
fprintf('Starting Pyff...\n'); pause(10);

pyff('init','HTMLViewer'); pause(2.5)
pyff('set', 'pages', questionnaires);
pyff('set', 'savedir', TODAY_DIR);

PrimaryScreenSize= get(0, 'ScreenSize');
screen_geometry= VP_SCREEN;
screen_geometry(2)= PrimaryScreenSize(4) - VP_SCREEN(4);
game_size= VP_SCREEN([3 4]);
game_size(2)= game_size(2)-100;
geometry= screen_geometry([1 2]) + ...
  (screen_geometry([3 4])-game_size)/2;
geometry([3 4])= game_size;

pyff('setint', 'geometry', geometry);

pyff('play');
stimutil_waitForInput('msg_next','when Questionnaire are completed.');
pyff('quit');
fprintf('Close Pyff window.\n')
