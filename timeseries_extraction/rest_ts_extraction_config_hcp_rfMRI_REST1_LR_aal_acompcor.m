% Parameter configuration for TS extraction for resting state fmri
% Parameter configuration for TS extraction for resting state fmri
% _________________________________________________________________________
% Stanford Cognitive and Systems Neuroscience Laboratory
% 
% $Id rest_ts_extraction_config.m $
% -------------------------------------------------------------------------

% Please specify parallel or nonparallel
paralist.parallel = '1';

% - Subject list
paralist.subjectlist = 'PROJECT_DIR/scripts/restfmri/extract_timeseries/subjectlist_missing_aal_rfMRI_REST1_LR.txt';

% - Run list
paralist.runlist = 'rfMRI_REST1_LR'; 

% - Project directory
paralist.projectdir = 'PROJECT_DIR/';

% - Preprocessed directory
paralist.preprocessed_dir = 'swgcar_spm12';
% - spm version
paralist.spmversion = 'spm12'; 
% - pipeline of processing
paralist.pipeline = 'swgcar';

% Please specify the ROI folders
paralist.roi_dir = 'SCRIPTS_DIR/atlases/aal/3D/'; %'ROI_DIR/';
% output directory
paralist.output_dir = 'PROJECT_DIR/data/imaging/timeseries/participant_level/acompcor/mat/aal/';

% Please specify the ROI list (full file name with extensions; ROI must be in .mat format)
%paralist.roi_list = 'SCRIPTS_DIR/atlases/aal/3D/aal_roi_list.txt';
paralist.roi_list = 'SCRATCH_DIR/atlases/aal/3D/aal_roi_list.txt';
paralist.roi_type = 'atlas'

paralist.physio_method='acompcor';
% Please specify the TR
paralist.TR = .72;

% summary function for time series extraction:
%  'mean', 'median', 'eigen1', 'wtmean'
paralist.sumfunc = 'mean'

% Filter the data?
% '1' for yes
% '2' for no

paralist.filter_data='1';
paralist.subjectlist="PROJECT_DIR/scripts/restfmri/extract_timeseries/subjectlist_complete";
paralist.subjectlist="PROJECT_DIR/scripts/restfmri/extract_timeseries/subjectlist_complete.txt";
