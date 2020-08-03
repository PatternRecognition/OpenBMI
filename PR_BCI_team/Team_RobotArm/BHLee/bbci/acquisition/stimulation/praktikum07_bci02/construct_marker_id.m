function id=construct_marker_id();
tone_keys=[0:11];
tone_key_mode=[1 2];

no=size(tone_keys,2);
tone_keys=[tone_keys;tone_keys];
tone_keys=tone_keys(:)';

tone_key_mode=repmat(tone_key_mode,1,12);


id=[tone_keys;tone_key_mode];
