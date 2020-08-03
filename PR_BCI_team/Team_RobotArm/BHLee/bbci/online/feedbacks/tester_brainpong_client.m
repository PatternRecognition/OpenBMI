clear opt
opt.client_player= 1;
opt.position= [1 279 1280 753];

global werbung werbung_opt
werbung= 1;
werbung_opt= [];
werbung_opt.pictures(1).image= [DATA_DIR 'images/cebit_brainpong_logos1.png'];
werbung_opt.pictures(2).image= [DATA_DIR 'images/cebit_brainpong_logos2.png'];


hnd= feedback_brainpong_client(gcf, opt);
hnd= hnd([2 3 5 6 9 11 14]);
set(hnd, 'Visible','on');
