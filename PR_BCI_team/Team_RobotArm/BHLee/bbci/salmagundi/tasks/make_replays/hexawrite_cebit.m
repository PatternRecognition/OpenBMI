setup_bbci_bet
unix('xset s off');
unix('xset -dpms');

opt_fb= struct('client_machines',[]);
opt_fb.position= [1 410 800 600];
opt_fb.text_spec= {'FontName','Courier'};
opt_movie= struct('freeze_out', 2);
opt_movie.fade_out= 1;
opt_movie.overwrite= 1;
opt= strukt('opt_fb',opt_fb, 'opt_movie',opt_movie);
opt.position= [1 410 800 600];
opt.force_set= {{63, 'FontName','Courier', 'FontSize',0.05}};

global LOG_DIR


sub_dir= 'Guido_06_03_09'
LOG_DIR= ['/home/neuro/data/BCI/bbciRaw/' sub_dir '/log/'];

  
logno= 173; opt.start= 1058.6; opt.stop= 1375;
replay('hexawrite', logno, opt, ...
       'save',sprintf('%s_hexawrite_%03d', sub_dir, logno));
%% DER_WUNSCH_IST_DER_VATER_DES_GEDANKEN
%% 37 characters in 311.9 sec: 7.1 char/min.


logno= 167; opt.start= 348.800; opt.stop= 998;
replay('hexawrite', logno, opt, ...
       'save',sprintf('%s_hexawrite_%03d', sub_dir, logno));
%% GROSSHIRNRINDE_AM_STEUER._DURCH_GEDANKENKRAFT._DAS_DENKEN_VERSTEHEN.
%% 68 characters in 644.8 sec: 6.3 char/min.

logno= 172; opt.start= 0; opt.stop= 737;
replay('hexawrite', logno, opt, ...
       'save',sprintf('%s_hexawrite_%03d', sub_dir, logno));
%% DIE_GEDANKEN_SIND_FREI._ICH_DENKE_ALSO_SCHREIBE_ICH.
%% 52 characters in 683.6 sec: 4.6 char/min.

logno= 176; opt.start= 160.080; opt.stop= 914;
replay('hexawrite', logno, opt, ...
       'save',sprintf('%s_hexawrite_%03d', sub_dir, logno));
%% DER_MENSCH_DENKT_DAS_HIRN_LENKT._DAS_PFERD_FRISST_KEINEN_GURKENSALAT
%% 68 characters in 749.6 sec: 5.4 char/min.

logno= 177; opt.start= 1205.920; opt.stop= 2140;
replay('hexawrite', logno, opt, ...
       'save',sprintf('%s_hexawrite_%03d', sub_dir, logno));
%% BERLIN_BRAIN_COMPUTER_INTERFACE._BERNSTEIN_ZENTRUM_BERLIN._ICH_BIN_EIN_CURSOR._HOLT_MICH_HIER_RAUS._
%% 100 characters in 929.5 sec: 6.5 char/min.

logno= 180; opt.start= 26.360; opt.stop= 888;
replay('hexawrite', logno, opt, ...
       'save',sprintf('%s_hexawrite_%03d', sub_dir, logno));
% BERLIN_BRAIN_COMPUTER_INTERFACE._CAN_YOU_IMAGINE?_MY_BRAIN_HURTS
% 64 characters in 857.5 sec: 4.5 char/min.

logno= 183; opt.start= 207.3; opt.stop= 834;
replay('hexawrite', logno, opt, ...
       'save',sprintf('%s_hexawrite_%03d', sub_dir, logno));
%% BERLIN_BRAIN_COMPUTER_INTERFACE._DAS_DENKEN_VERSTEHEN._PAUSE?
%% 61 characters in 626.7 sec: 5.7 char/min.

logno= 185; opt.start= 54.160; opt.stop= 528;
replay('hexawrite', logno, opt, ...
       'save',sprintf('%s_hexawrite_%03d', sub_dir, logno));
%% SPITZENFORSCHUNG_GEFOERDERT_VOM_BUNDESMINISTERIUM
%% 50 characters in 470.4 sec: 6.3 char/min.

logno= 189; opt.start= 52.400; opt.stop= 838;
replay('hexawrite', logno, opt, ...
       'save',sprintf('%s_hexawrite_%03d', sub_dir, logno));
%% MENTALE_SCHREIBMASCHINE_IN_AKTION._BERLIN_BRAIN_COMPUTER_INTERFACE
%% 66 characters in 781.3 sec: 5.1 char/min.




sub_dir= 'Guido_06_03_10'
LOG_DIR= ['/home/neuro/data/BCI/bbciRaw/' sub_dir '/log/'];


logno= 195; opt.start= 249.200; opt.stop= 753;
replay('hexawrite', logno, opt, ...
       'save',sprintf('%s_hexawrite_%03d', sub_dir, logno));
%% BERLIN_BRAIN_COMPUTER_INTERFACE._DAS_DENKEN_VERSTEHEN.
%% 54 characters in 499.6 sec: 6.5 char/min.

logno= 196; opt.start= 1229.000; opt.stop= 1836;
replay('hexawrite', logno, opt, ...
       'save',sprintf('%s_hexawrite_%03d', sub_dir, logno));
%% STEUERUNG_DURCH_GEDANKENKRAFT._DIE_GEDANKEN_SIND_FREI.
%% 54 characters in 602.6 sec: 5.4 char/min.

logno= 197; opt.start= 2572.400; opt.stop= 3035;
replay('hexawrite', logno, opt, ...
       'save',sprintf('%s_hexawrite_%03d', sub_dir, logno));
%% UND_MIT_GEISTESSTAERKE_TU_ICH_WUNDER_AUCH.
%% 42 characters in 458.3 sec: 5.5 char/min.

logno= 198; opt.start= 4147.720; opt.stop= 4648;
replay('hexawrite', logno, opt, ...
       'save',sprintf('%s_hexawrite_%03d', sub_dir, logno));
%% MENTALE_MENTALE_SCHREIBMASCHINE_ENTWICKELT_VOM_BBCI_TEAM.
%% 57 characters in 496.4 sec: 6.9 char/min.

logno= 202; opt.start= 2929.9; opt.stop= 3666;
replay('hexawrite', logno, opt, ...
       'save',sprintf('%s_hexawrite_%03d', sub_dir, logno));
%% DAS_PFERD_FRISST_KEINEN_GURKENSALAT._BERLIN_BRAIN_COMPUTER_INTERFACE.
%% 69 characters in 734.1 sec: 5.6 char/min.

logno= 205; opt.start= 5402.440; opt.stop= 6217;
replay('hexawrite', logno, opt, ...
       'save',sprintf('%s_hexawrite_%03d', sub_dir, logno));
%% BMBF_MENSCH_TECHNIK_INTERAKTION._FRAUNHOFER_FIRST_UND_CHARITE_BERLIN_BBCI
%% 73 characters in 810.6 sec: 5.4 char/min.

logno= 206; opt.start= 6764.080; opt.stop= 7686;
replay('hexawrite', logno, opt, ...
       'save',sprintf('%s_hexawrite_%03d', sub_dir, logno));
%% WORUEBER_MAN_NICHT_SPRECHEN_KANN_DARUEBER_SOLL_MAN_SCHWEIGEN.
%% 61 characters in 918.1 sec: 4.0 char/min.

logno= 207; opt.start= 8904.680; opt.stop= 9453;
replay('hexawrite', logno, opt, ...
       'save',sprintf('%s_hexawrite_%03d', sub_dir, logno));
%% DER_KOPF_IST_RUND_DAMIT_DAS_DENKEN_DIE_RICHTUNG_WECHSELN_KANN
%% 61 characters in 544.6 sec: 6.7 char/min.

logno= 208; opt.start= 13.480; opt.stop= 656;
replay('hexawrite', logno, opt, ...
       'save',sprintf('%s_hexawrite_%03d', sub_dir, logno));
%% ICH_BIN_MUEDE._FEIERABEND?_BERLIN_BRAIN_COMPUTER_INTERFACE
%% 58 characters in 638.3 sec: 5.5 char/min.





sub_dir= 'Michael_06_03_09'
LOG_DIR= ['/home/neuro/data/BCI/bbciRaw/' sub_dir '/log/'];

logno= 127; opt.start= 9.040; opt.stop= 395;
replay('hexawrite', logno, opt, ...
       'save',sprintf('%s_hexawrite_%03d', sub_dir, logno));
%% ICH_BIN_EIN_VORDENKER
%% 21 characters in 381.5 sec: 3.3 char/min.

logno= 130; opt.start= 5.9; opt.stop= 409;
replay('hexawrite', logno, opt, ...
       'save',sprintf('%s_hexawrite_%03d', sub_dir, logno));
%% ICH_BIN_EIN_VORDENKER
%% 21 characters in 403 sec: 3.2 char/min.

logno= 132; opt.start= 9.400; opt.stop= 1301;
replay('hexawrite', logno, opt, ...
       'save',sprintf('%s_hexawrite_%03d', sub_dir, logno));
%% ICH_BIN_EIN_VORDENKER._DIE_MENTALE_SCHREIBMASCHINE_IN_AKTION
%% 60 characters in 1259.4 sec: 2.9 char/min.

logno= 133; opt.start= 3.2; opt.stop= 695;
replay('hexawrite', logno, opt, ...
       'save',sprintf('%s_hexawrite_%03d', sub_dir, logno));
%% HALLO_SUESSE.
%% 13 characters in 157.1 sec: 5.0 char/min.

logno= 134; opt.start= 5.160; opt.stop= 404;
replay('hexawrite', logno, opt, ...
       'save',sprintf('%s_hexawrite_%03d', sub_dir, logno));
%% ICH_BIN_EIN_VORDENKER
%% 21 characters in 394.4 sec: 3.2 char/min.

logno= 135; opt.start= 15.000; opt.stop= 481;
replay('hexawrite', logno, opt, ...
       'save',sprintf('%s_hexawrite_%03d', sub_dir, logno));
%% ICH_BIN_EIN_VORDENKER.
%% 22 characters in 461.7 sec: 2.9 char/min.

logno= 136; opt.start= 585.720; opt.stop= 1021;
replay('hexawrite', logno, opt, ...
       'save',sprintf('%s_hexawrite_%03d', sub_dir, logno));
%% THE_GADGET_SHOW
%% 15 characters in 431.0 sec: 2.1 char/min.

logno= 138; opt.start= 9.160; opt.stop= 617;
replay('hexawrite', logno, opt, ...
       'save',sprintf('%s_hexawrite_%03d', sub_dir, logno));
%% NEUES_AUF_DREISAT
%% 17 characters in 603.7 sec: 1.7 char/min.




sub_dir= 'Michael_06_03_10'
LOG_DIR= ['/home/neuro/data/BCI/bbciRaw/' sub_dir '/log/'];


logno= 148; opt.start= 46.920; opt.stop= 1110;
replay('hexawrite', logno, opt, ...
       'save',sprintf('%s_hexawrite_%03d', sub_dir, logno));
%% ICH_BIN_EIN_BERLINER._MENS_SANA_IN_CAMPARI_SODA
%% 47 characters in 1058.6 sec: 2.7 char/min.

logno= 149; opt.start= 1933.480; opt.stop= 2448;
replay('hexawrite', logno, opt, ...
       'save',sprintf('%s_hexawrite_%03d', sub_dir, logno));
%% ICH_DENKE_ALSO_BIN_ICH?
%% 23 characters in 510.2 sec: 2.7 char/min.

logno= 150; opt.start= 3258; opt.stop= 3826;
replay('hexawrite', logno, opt, ...
       'save',sprintf('%s_hexawrite_%03d', sub_dir, logno));
%% IN_GEDANKEN_SCHON_VIEL_WEITER.
%% 30 characters in 562.8 sec: 3.2 char/min.

logno= 151; % EMPTY

logno= 152; opt.start= 112.520; opt.stop= 354;
replay('hexawrite', logno, opt, ...
       'save',sprintf('%s_hexawrite_%03d', sub_dir, logno));
%% HALLO_SEBI.

logno= 153; opt.start= 457.880; opt.stop= 3902;
replay('hexawrite', logno, opt, ...
       'save',sprintf('%s_hexawrite_%03d', sub_dir, logno));
%% WO_EIN_BEGEISTERTER_STEHT_IST_DER_GIPFEL_DER_WELT._E.DORFF
%% 58 characters 

logno= 154; opt.start= 4371.440; opt.stop= 4738;
replay('hexawrite', logno, opt, ...
       'save',sprintf('%s_hexawrite_%03d', sub_dir, logno));
%% DER_GEIST_IST_KEIN_ELEFANT.
%% 27 characters in 366 sec: 4.4 char/min.

logno= 155; opt.start= 5976.560; opt.stop= 5347.5;
replay('hexawrite', logno, opt, ...
       'save',sprintf('%s_hexawrite_%03d', sub_dir, logno));
%% DIE_GEDANKEN_SIND_FREI

%logno= 156; opt.start= 6516.000; opt.stop= 6690;
%replay('hexawrite', logno, opt, ...
%       'save',sprintf('%s_hexawrite_%03d', sub_dir, logno));
%% MASSIMO
%% 7 characters in 176.6 sec: 2.4 char/min.

logno= 157; opt.start= 7892.320; opt.stop= 8755;
replay('hexawrite', logno, opt, ...
       'save',sprintf('%s_hexawrite_%03d', sub_dir, logno));
%% ICH_BIN_EIN_FRAUNHOFER_VORDENKER.
%% 33 characters in 858.5 sec: 2.3 char/min.

logno= 158; opt.start= 9961.560; opt.stop= 10556;
replay('hexawrite', logno, opt, ...
       'save',sprintf('%s_hexawrite_%03d', sub_dir, logno));
%% MIT_UNS_KOENNEN_SIE_RECHNEN.
%% 28 characters in 590.0 sec: 2.8 char/min.

logno= 159; % EMPTY

logno= 160; opt.start= 13.640; opt.stop= 446;
replay('hexawrite', logno, opt, ...
       'save',sprintf('%s_hexawrite_%03d', sub_dir, logno));
%% DIE_SONNE_IST_VON_KUPFER._DAS_PFER
%% 24 characters in 432.4 sec: 3.3 char/min.
