function generate_images()
    addpath(genpath('TOOLBOX_DIR/BrainNetViewer_20191031/'))
    addpath(genpath('TOOLBOX_DIR/vistasoft/'))

    niis_dir='PROJECT_DIR/results/restfmri/dnn/niis/indiv_niis/';
    output_dir='PROJECT_DIR/results/restfmri/dnn/brain_plots/indiv_plots/';
    file_list=dir(niis_dir);

    for nii_idx=3:size(file_list,1)
            nii_name=[niis_dir file_list(nii_idx).name];
            png_name=[output_dir file_list(nii_idx).name '.png'] ;

            data = niftiRead(nii_name);
            
            load('default_settings_afnipos.mat');
            EC.vol.px=max(data.data(data.data>0));
%            %EC.vol.px=min(data.data(data.data>0));
%            EC.vol.display=1;
%            EC.vol.nx=min(data.data,[],'all');
%            %EC.vol.nx=-max(data.data(data.data>0));
%            EC.vol.px=max(data.data,[],'all');
%            EC.vol.color_map=13;
            %EC.vol.adjustCM =1;
%            disp([png_name '.mat'])
            save([png_name '.mat'],'EC');
            
            surface_file=['merged_fs_surf.nv'];
            settings_file=[png_name '.mat'];
            png_output=[png_name];
            
            BrainNet_MapCfg(surface_file,...
            nii_name,...
            settings_file,...
            png_name);

    end
end
