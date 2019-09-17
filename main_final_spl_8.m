% Main denoising procedure
% Input: shapename, and gaussian noise paramter
% Parameter: sigma for weighting tuning, Knn
% Output: denoised PCD
% After talk with Prof. Shi, replace I with W --- but use I for iteration 2
% gamma --- but small gamma for iteration 2
% comparing patches
% step1: do denoising with some diffusion filter (slightly), make smooth (lie on manifold) and uniform
% step2: project to a plane, do the mapping and compute the distance
% step3: do interpolation with griddata
% step4: compute W
% Jin Zeng, 20171018
warning('off');
% close all; clc
% version: result9 dn3
addpath './tool';
addpath './metric';
addpath './data';
addpath './ulli';
noise = [0.03, 0.05, 0.07, 0.1];
shape_files = [600, 1200, 1200, 1500, 0];
%% todo
for shape = 1 % soldier
    switch shape
        case 1
            shapename = 'soldier';
        case 2
            shapename = 'longdress';
        case 3
            shapename = 'loot';
        case 4
            shapename = 'redandblack';
        case 5
            shapename = 'Frame_0141';
    end
    for noisetype = 4:4
        for file_e = 1:6
            file = file_e + shape_files(shape);
            %shapenam e = 'gargoyle';%dc,gargoyle,anchor,daratech,lordquas
            gt_filename = ['./data/', shapename , num2str(file), '_ds.ply'];
            % noisename = [shapename,'_gaussian_noise_',num2str(0.03)];%0.03,0.04
            n_filename = ['./data/', shapename , num2str(file), 'ds_gaussian_noise_', num2str(noise(noisetype)), '.ply'];
            n_filename_ne = ['./data/', shapename , num2str(file), 'ds_gaussian_noise_', num2str(noise(noisetype))];
            if file_e == 1
                np_filename = gt_filename; %empty file
            else
                np_filename = ['./data/', shapename , num2str(file-1), 'ds_gaussian_noise_', num2str(noise(noisetype)), '_dn3.ply'];% pre frame
            end
            % nn_filename = ['soldier_vox10_0601_ds_gaussian_noise_0.05.ply']; % next frame
            
            %% read in file for ground-truth and noisy model
            pt_gt = pcread(gt_filename);
            X_gt = pt_gt.Location;
            pt_X= pcread(n_filename);
            X = pt_X.Location;
            pt_p = pcread(np_filename);
            X_p = pt_p.Location;
            pt_p = pointCloud(X_p);
            % pt_n = pcread(nn_filename);
            % X_n = pt_n.Location;
            % pt_n = pointCloud(X_n);
            temp = meandistance(X_gt, X);
            disp(temp);
            fid = fopen('result9.txt', 'a');
            fprintf(fid, 'initial mse %s %d %.2f %.4f\r\n', shapename, file, noise(noisetype), temp);
            fclose(fid);
            %
            % disp(meandensity(X_gt))
            %% todo end
            
            %% parameter settings
            max_itr = 4; change_tolerance = 0.0001; % stop criteria
            % choose fps for uniform sampling
            flag_sample = 1; SAMPLING_SET = ceil(0.5*size(X,1));
            % if not, use grid sample
            gridStep = 0.7;
            % patch size, searching window size,
            pk = 30; wink = 16;
            % graph construction parameter: epsilon, patch similarity threshold
            eps1 = 0.45*ones(max_itr,1); eps1(1:2)=[1.05, 0.75]; eps1 = eps1.^2;
            threshold_d = 20;
            if noisetype ~= 2 % noise variance = 0.05
                threshold_d = 20;
            end
            % optimization parameter
            gamma=0.5; lambda1 = 30*ones(max_itr,1); lambda2=zeros(max_itr, 1);
            if (noisetype == 1)
                lambda1 = [7, 40, 70, 300];
                if file_e ~= 1
                    lambda2(1:2) = [0.4, 0];
                end
            end
            if (noisetype == 2)
                lambda1 = [1.1, 10, 17, 30];
                if file_e ~= 1
                    lambda2 = [28, 0.1 , 0, 0];
                end
            end
            if (noisetype == 3)
                lambda1 = [1, 4, 12, 50];
                if file_e ~= 1
                    lambda2 = [270, 0, 0, 0];
                end
            end
            if (noisetype == 4)
                lambda1 = [0.4, 2.1, 6, 12];
                if file_e ~= 1
                    lambda2 = [3000, 0, 0, 0];
                end
            end
            % lambda2 = [0 0 0 0]; % baseline
            %% todo
            winf = 10; % window to find most similar patch in adjcent frames
            %% todo end
            
            %% pixel graph (no prefiltering now)
            X_m = X;
            
            %% patch center, patch construction with size=pk, and patch graph with k=wink
            pt_pre = pointCloud(X);
            % patch center
            if flag_sample
                srf = struct('X',X(:,1),'Y',X(:,2),'Z',X(:,3));
                ifps = fps_euc(srf,SAMPLING_SET);
                pt_C = pointCloud(X(ifps,:));
            else
                pt_C = pcdownsample(pt_pre,'gridAverage',gridStep);
            end
            % patch construction
            pn = pt_C.Count; % patch number
            P = zeros(pn,pk); % patch center and the node indices in the patch
            for i = 1:pn
                [indices,dists] = findNearestNeighbors(pt_pre,pt_C.Location(i,:),pk);
                P(i,:) = indices;
            end
            Nf = pn*pk; % node graph size
            % patch graph
            P_win = zeros(pn, wink);
            P_winf = zeros(pn, winf);
            for i = 1:pn
                [indices,dists] = findNearestNeighbors(pt_C,pt_C.Location(i,:),wink);
                P_win(i,:) = indices;

            end
            N_p = pt_p.Count; % pre frame point number
            P_p_win = zeros(N_p, pk);
            for i = 1:N_p
                [indices,dists] = findNearestNeighbors(pt_p,pt_p.Location(i,:),pk);
                P_p_win(i,:) = indices;
            end
            P_win = P_win(:,2:end);
            
            %% Denoising
            dM = zeros(max_itr,1); dH = dM;
            X_pre = X;
            % set center
            X_c = zeros(pn,3);
            for c = 1:3
                u = X_pre(:,c);
                X_c(:,c) = mean(u(P),2);
            end
            for i = 1:pn
                [indices,dists] = findNearestNeighbors(pt_p,X_c(i,:),winf);
                P_winf(i,:) = indices;
            end
            
            for itr = 1:max_itr
                %% prepare reference frame for each patch: so as to compute height field and projection
                % shift the point wrt the center
                f = zeros(pn,pk,3); fo = f;
                for c = 1:3
                    u = X_m(:,c);
                    f(:,:,c) = u(P)-repmat(X_c(:,c),1,pk);
                    u = X_pre(:,c);
                    fo(:,:,c) = u(P)-repmat(X_c(:,c),1,pk);
                end
                % normal at each center
                fn = zeros(pn,3); ptmp = zeros(pk,3);
                for i = 1:pn
                    ptmp(:,:) = f(i,:,:);
                    cov = ptmp'*ptmp;
                    [Vtmp,Dtmp] = eig(cov);
                    fn(i,:) = Vtmp(:,1); % reference plane with normal vertor stored in fn
                end
                %% todo
                P_p = zeros(pk*pn, 3);
                P_n = zeros(pk*pn, 3);
                %% end todo
                
                %% patch similarity and connection
                D_i = zeros(Nf,wink-1);D_w = zeros(Nf,wink-1); tmp_p1 = (1:pk)';
                D_wp = zeros(Nf, 1);
                % temp variable for comparing fixed and moving patches
                fixed = zeros(pk,3); fixed_n = zeros(3,1); moving = fixed;
                for i = 1:pn
                    %         if mod(i, 1000) == 0
                    %             disp(['i=', num2str(i)]);
                    %         end
                    % load fixed
                    fixed(:,:) = f(i,:,:); fixed_n(:) = fn(i,:);
                    % compute height field, and projection on reference plane
                    hf = fixed*fixed_n; fixed_proj = fixed-hf*fixed_n';
                    % search the neighbors of fixed
                    wedge = P_win(i,:);
                    for j = 1:length(wedge)
                        % load moving
                        w = wedge(j); moving(:,:) = f(w,:,:);
                        % height field and projection on plane
                        hm = moving*fixed_n; moving_proj = moving-hm*fixed_n';
                        % use projection for neighbor search
                        xx = sum((fixed_proj.^2),2);
                        yy = sum((moving_proj.^2),2);
                        xy = fixed_proj*moving_proj';
                        DistMat = repmat(xx,1,length(yy))-2*xy+repmat(yy',length(xx),1);
                        [dmin, idx] = min(DistMat,[],2);
                        % find interpolation nodes
                        preserve_id = dmin<=threshold_d;
                        discard_id = dmin>threshold_d;
                        Dsum = 0;
                        if sum(discard_id)
                            tempid = find(discard_id);
                            if sum(discard_id) > 1
                                [~, IDistMat] = sort(DistMat(discard_id,:),2);
                                for ii = 1:length(tempid)
                                    tmp_pt = tempid(ii);
                                    dx = interpolate_plane(moving(IDistMat(ii,1),:),moving(IDistMat(ii,2),:),moving(IDistMat(ii,3),:),fixed(tmp_pt,:),fixed_n');
                                    Dsum = Dsum+dx^2;
                                end
                            else
                                [~, IDistMat] = sort(DistMat(discard_id,:));
                                dx = interpolate_plane(moving(IDistMat(1),:),moving(IDistMat(2),:),moving(IDistMat(3),:),fixed(tempid,:),fixed_n');
                                Dsum = Dsum+dx^2;
                            end
                        end
                        % use height field for computing distance
                        Dist = (sum((hm(idx(preserve_id))-hf(preserve_id)).^2)+Dsum)/pk;
                        %% todo
                        %             D_w(pk*(i-1)+tmp_p1,j) = Dist;
                        %             D_i(pk*(i-1)+tmp_p1,j) = pk*(w-1)+idx(:);
                        D_w(pk*(i-1)+tmp_p1,j)=Dist;
                        D_i(pk*(i-1)+tmp_p1,j) = pk*(w-1)+idx(:);
                        %% todo end
                    end
                    
                    
                    %% todo
                    % find most similar patch in pre frame
                    pt_p = pointCloud(X_p);
                    indices = P_winf(i,:);
                    Dist_min = 999999;
                    for j = 1:length(indices)
                        % build patch
                        indices_p = P_p_win(indices(j),:);
                        center = zeros(1,3);
                        for c=1:3
                            center(c) = mean(X_p(indices_p,c));
                        end
                        moving = X_p(indices_p,:) - repmat(center, pk, 1);
                        % height field and projection on plane
                        hm = moving*fixed_n; moving_proj = moving-hm*fixed_n';
                        % use projection for neighbor search
                        xx = sum((fixed_proj.^2),2);
                        yy = sum((moving_proj.^2),2);
                        xy = fixed_proj*moving_proj';
                        DistMat = repmat(xx,1,length(yy))-2*xy+repmat(yy',length(xx),1);
                        [dmin, idx] = min(DistMat,[],2);
                        % find interpolation nodes
                        preserve_id = dmin<=threshold_d;
                        discard_id = dmin>threshold_d;
                        Dsum = 0;
                        if sum(discard_id)
                            tempid = find(discard_id);
                            if sum(discard_id) > 1
                                [~, IDistMat] = sort(DistMat(discard_id,:),2);
                                for ii = 1:length(tempid)
                                    tmp_pt = tempid(ii);
                                    dx = interpolate_plane(moving(IDistMat(ii,1),:),moving(IDistMat(ii,2),:),moving(IDistMat(ii,3),:),fixed(tmp_pt,:),fixed_n');
                                    Dsum = Dsum+dx^2;
                                end
                            else
                                [~, IDistMat] = sort(DistMat(discard_id,:));
                                dx = interpolate_plane(moving(IDistMat(1),:),moving(IDistMat(2),:),moving(IDistMat(3),:),fixed(tempid,:),fixed_n');
                                Dsum = Dsum+dx^2;
                            end
                        end
                        % use height field for computing distance
                        Dist = (sum((hm(idx(preserve_id))-hf(preserve_id)).^2)+Dsum)/pk;
                        if Dist < Dist_min
                            Dist_min = Dist;
                            D_wp(pk*(i-1)+tmp_p1) = Dist;
                            P_p(pk*(i-1)+tmp_p1, :) = moving(idx(:), :);
                        end
                    end
                    
                end
                tmp = repmat((1:Nf)',1,wink-1);
                D_p = sparse(tmp(:),D_i(:),D_w(:),Nf,Nf);
                D_pp = sparse(tmp(:,1), tmp(:,1), D_wp(:), Nf, Nf);
                D_pp(D_pp>0) = exp(-D_pp(D_pp>0)./(2*eps1(itr)));
                %% laplacian construction
                %  mean_p = sum(sum(D_p))/sum(sum(D_p>0));
                A = D_p;
                if noisetype == 1
                    A(A>5) = 0;
                else
                    if noisetype == 3
                        A(A>10) = 0;
                    elseif noisetype == 4
                        A(A>30) = 0;
                    end
                end
                %     mask1 = logical(tril(A, -1)); mask2 = logical(triu(A, 1));
                %     mask = mask1'.*mask2; mask = mask+mask;
                %     A = A.*mask; A = 0.5*(A+A');
                mask1 = logical(tril(A));
                A = A-mask1'.*A;
                A = A+A';
                A(A>0)=exp(-A(A>0)./(2*eps1(itr)));
                weight_rec=diag(sum(A).^(-gamma));
                A=weight_rec*A*weight_rec;
                Dn=sum(A);
                Dn=diag(Dn);
                L=Dn-A;
                
                %% optimize
                %% todo
                %     L_tt = L(pn*pk+1:2*pn*pk,pn*pk+1:2*pn*pk);
                %     L_tp = L(pn*pk+1:2*pn*pk,1:pn*pk);
                %     L_tn = L(pn*pk+1:2*pn*pk,2*pn*pk+1:3*pn*pk);
                X_rec = zeros(length(X_pre),3);
                tmp = P';
                U = sparse(1:Nf,tmp(:),ones(Nf,1),Nf,length(X_pre));
                
                
                for c = 1:3
                    t = repmat(X_c(:,c),1,pk);
                    t = t'; t = t(:);
                    u0 = X_pre(:,c);
                    %
                    C = U'*L*t+lambda1(itr)*u0+lambda2(itr)*U'*D_pp*(t+P_p(:,c));
                    B = U'*L*U+lambda1(itr)*speye(length(X_pre))+lambda2(itr)*U'*D_pp*U;
                    %                     C = U'*L*t+lambda1(itr)*u0;
                    %                     B = U'*L*U+lambda1(itr)*speye(length(X_pre));
                    X_rec(:,c) = lsqr(B,double(C),1e-06,10000);
                    
                end
                
                % write_ply_only_points(X_rec,[n_filename_ne,'_hhh_',num2str(itr),'_',num2str(lambda1(itr)),'_' num2str(lambda2(itr)),'.ply']);
                mset = meandistance(X_gt, X_rec);
                %                 % dH(itr,1) = hausdorff(X_gt, X_rec);
                %                 change_dM = meandistance(X_pre, X_rec);
                % disp([num2str(itr), ',', num2str(lambda1(itr)), ',', num2str(lambda2(itr)), ',' ,num2str(dM(itr,1)) ', ' num2str(change_dM)]);
                fid = fopen('result9.txt', 'a');
                fprintf(fid, '%s %d %.2f %d %.4f\r\n', shapename, file, noise(noisetype), itr, mset);
                fclose(fid);
                %                 if change_dM < change_tolerance % terminate if it doesn't change too much
                %                     break;
                %                 end
                
                
                %     end
                X_pre = X_rec;
                X_m = X_pre;
            end
            mdc = meandistance(X_gt, X_rec);
            write_ply_only_points(X_rec,[n_filename_ne '_dn3.ply']);
            disp(['meandistance=',num2str(mdc)]);
            fid = fopen('result9.txt', 'a');
            fprintf(fid, 'final mse %s %d %.2f %.4f\r\n', shapename, file, noise(noisetype), mdc);
            fclose(fid);
        end
    end
end
